#!/usr/bin/env python3
"""
Batch ipSAE summarizer for AlphaFold3 and Boltz-2 runs.

Features
- Discovers AF3 or Boltz-2 runs in a root directory.
- For AF3: each run in its own subdirectory with full_data_<i>.json + model_<i>.cif pairs.
- For Boltz-2: navigates deeper folder structure to find pae_*.npz + *.cif pairs.
- Runs ipsae.py for each pair (in parallel), unless output exists or --force.
- Extracts ipSAE from asym rows oriented binder->target (default B->A for AF3, A->target chain for Boltz).
- Aggregates per-run (binder=subdir name): mean and stdev ipSAE.
- Writes a single CSV summary.

Example (AlphaFold 3)
  python summarize_ipsae.py \
    --root /path/to/Adaptyv-Submissions \
    --source af3 \
    --binder-chains B \
    --target-chains A \
    --pae 10 --dist 10 \
    --workers 8 \
    --out /path/to/Adaptyv-Submissions/ipsae_summary.csv

Example (Boltz-2)
  python summarize_ipsae.py \
    --root /path/to/boltz-predictions \
    --source boltz \
    --prediction-type target \
    --binder-chains A \
    --target-chains TA \
    --pae 15 --dist 15 \
    --workers 8 \
    --out /path/to/boltz-predictions/ipsae_summary.csv
"""
import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ModelTask:
    binder_name: str
    json_path: str  # PAE file: .json for AF3, .npz for Boltz
    cif_path: str   # Structure file: .cif or .pdb
    pae_cutoff: int
    dist_cutoff: int
    ipsae_py: str
    force: bool
    binder_chains: Tuple[str, ...]
    target_chains: Tuple[str, ...]
    pair_reduction: str  # "max" or "mean"


@dataclass
class ModelResult:
    binder_name: str
    ipsae_value: Optional[float]  # None if not found or failed
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ipSAE across many AF3 or Boltz-2 runs.")
    parser.add_argument("--root", required=True, help="Root directory containing many run subdirectories.")
    parser.add_argument(
        "--source",
        choices=("af3", "boltz"),
        default="af3",
        help='Source type: "af3" for AlphaFold 3, "boltz" for Boltz-2 (default: af3).',
    )
    parser.add_argument(
        "--prediction-type",
        default=None,
        help='For Boltz-2: filter predictions by type (e.g., "target", "self", "antitarget"). '
             'If not specified, processes all prediction types.',
    )
    parser.add_argument(
        "--ipsae",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ipsae.py"),
        help="Path to ipsae.py (default: ipsae.py in this repo).",
    )
    parser.add_argument("--pae", type=int, default=10, help="PAE cutoff (max 10 for AF3, typically 15 for Boltz).")
    parser.add_argument("--dist", type=int, default=10, help="Distance cutoff (default 10).")
    parser.add_argument(
        "--binder-chains",
        default="B",
        help='Comma-separated binder chain IDs (default "B"). For Boltz-2, typically "A".',
    )
    parser.add_argument(
        "--target-chains",
        default="A",
        help='Comma-separated target chain IDs (default "A"). For Boltz-2, may be "TA" or similar.',
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of parallel workers.")
    parser.add_argument("--force", action="store_true", help="Recompute even if per-model ipSAE txt exists.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: <root>/ipsae_summary.csv).",
    )
    parser.add_argument(
        "--pair-reduction",
        choices=("max", "mean"),
        default="max",
        help='How to reduce multiple binder->target chain-pair ipSAEs in a model (default "max").',
    )
    return parser.parse_args()


def clamp_pae(pae_value: int, source: str) -> int:
    # For AF3, PAE cutoff max is 10; for Boltz, allow higher values
    if source == "af3":
        return min(int(pae_value), 10)
    return int(pae_value)


def to_chain_tuple(value: str) -> Tuple[str, ...]:
    raw = [x.strip().upper() for x in value.split(",") if x.strip()]
    return tuple(raw)


def discover_run_subdirs(root_dir: str) -> List[str]:
    try:
        entries = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
    except FileNotFoundError:
        raise SystemExit(f"Root directory not found: {root_dir}")
    return [p for p in entries if os.path.isdir(p)]


# ============================================================================
# AlphaFold 3 discovery functions
# ============================================================================

def find_af3_model_pairs(run_dir: str) -> List[Tuple[str, str]]:
    """
    Returns list of (full_data_json, model_cif) pairs for indices present in run_dir.
    Matching rule: replace '_full_data_<i>.json' with '_model_<i>.cif'.
    """
    pairs: List[Tuple[str, str]] = []
    for name in os.listdir(run_dir):
        if not name.endswith(".json"):
            continue
        if "_full_data_" not in name:
            continue
        m = re.search(r"_full_data_(\d+)\.json$", name)
        if not m:
            continue
        idx = m.group(1)
        json_path = os.path.join(run_dir, name)
        cif_name = name.replace(f"_full_data_{idx}.json", f"_model_{idx}.cif")
        cif_path = os.path.join(run_dir, cif_name)
        if os.path.exists(cif_path):
            pairs.append((json_path, cif_path))
    # Sort by index implied in file to keep deterministic ordering
    pairs.sort(key=lambda t: t[0])
    return pairs


# ============================================================================
# Boltz-2 discovery functions
# ============================================================================

def find_boltz_prediction_dirs(run_dir: str, prediction_type: Optional[str] = None) -> List[str]:
    """
    Finds all Boltz-2 prediction directories within a run directory.
    
    Structure: <run_dir>/outputs/boltz_results_<name>_vs_<type>/predictions/<name>/
    
    Args:
        run_dir: Path to a binder directory (e.g., binder_NiVG-BG-VHH-2_0043)
        prediction_type: Optional filter like "target", "self", "antitarget"
                        Matches against "_vs_<type>" in the path.
    
    Returns:
        List of paths to prediction directories containing model files.
    """
    prediction_dirs: List[str] = []
    outputs_dir = os.path.join(run_dir, "outputs")
    
    if not os.path.isdir(outputs_dir):
        return prediction_dirs
    
    for boltz_results_name in os.listdir(outputs_dir):
        if not boltz_results_name.startswith("boltz_results_"):
            continue
        
        # Filter by prediction type if specified
        if prediction_type:
            # Match _vs_<type> pattern (e.g., _vs_target, _vs_self)
            type_pattern = f"_vs_{prediction_type}"
            # Also match partial: _vs_target_nipah_g should match "target"
            if type_pattern not in boltz_results_name.lower() and f"_vs_{prediction_type}_" not in boltz_results_name.lower():
                continue
        
        predictions_base = os.path.join(outputs_dir, boltz_results_name, "predictions")
        if not os.path.isdir(predictions_base):
            continue
        
        # There's typically one subdirectory under predictions/
        for pred_subdir in os.listdir(predictions_base):
            pred_path = os.path.join(predictions_base, pred_subdir)
            if os.path.isdir(pred_path):
                prediction_dirs.append(pred_path)
    
    return prediction_dirs


def find_boltz_model_pairs(prediction_dir: str) -> List[Tuple[str, str]]:
    """
    Returns list of (pae_npz, model_cif) pairs for Boltz-2 predictions.
    
    Matching rule: 
        pae_<name>_model_<i>.npz -> <name>_model_<i>.cif
    """
    pairs: List[Tuple[str, str]] = []
    
    try:
        files = os.listdir(prediction_dir)
    except FileNotFoundError:
        return pairs
    
    # Find all PAE files
    pae_files = [f for f in files if f.startswith("pae_") and f.endswith(".npz")]
    
    for pae_file in pae_files:
        # Extract model name: pae_<name>_model_<i>.npz -> <name>_model_<i>.cif
        # Remove 'pae_' prefix and '.npz' suffix
        pae_stem = pae_file[4:-4]  # Remove 'pae_' and '.npz'
        cif_name = pae_stem + ".cif"
        
        pae_path = os.path.join(prediction_dir, pae_file)
        cif_path = os.path.join(prediction_dir, cif_name)
        
        if os.path.exists(cif_path):
            pairs.append((pae_path, cif_path))
    
    # Sort by model index to keep deterministic ordering
    pairs.sort(key=lambda t: t[0])
    return pairs


# ============================================================================
# Common functions
# ============================================================================

def ipsae_output_txt_path(struct_path: str, pae_cutoff: int, dist_cutoff: int) -> str:
    """Generate expected output txt path from structure file path."""
    # Remove extension (.cif or .pdb)
    if struct_path.endswith(".cif"):
        stem = struct_path[:-4]
    elif struct_path.endswith(".pdb"):
        stem = struct_path[:-4]
    else:
        stem = struct_path
    
    pae_str = f"{int(pae_cutoff):d}"
    if int(pae_cutoff) < 10:
        pae_str = "0" + pae_str
    dist_str = f"{int(dist_cutoff):d}"
    if int(dist_cutoff) < 10:
        dist_str = "0" + dist_str
    return f"{stem}_{pae_str}_{dist_str}.txt"


def run_ipsae_if_needed(task: ModelTask) -> Optional[str]:
    """
    Runs ipsae.py for the given model if output txt is missing or --force is set.
    Returns path to the per-model ipSAE .txt file on success, or None on failure.
    """
    output_txt = ipsae_output_txt_path(task.cif_path, task.pae_cutoff, task.dist_cutoff)
    if os.path.exists(output_txt) and not task.force:
        return output_txt

    if not os.path.exists(task.ipsae_py):
        sys.stderr.write(f"[WARN] ipsae.py not found: {task.ipsae_py}\n")
        return None

    cmd = [
        sys.executable,
        task.ipsae_py,
        task.json_path,
        task.cif_path,
        str(task.pae_cutoff),
        str(task.dist_cutoff),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(
            f"[ERR] ipsae.py failed for {task.cif_path}\n"
            f"      cmd: {' '.join(cmd)}\n"
            f"      rc: {e.returncode}\n"
            f"      stdout: {e.stdout.decode(errors='ignore')}\n"
            f"      stderr: {e.stderr.decode(errors='ignore')}\n"
        )
        return None

    if os.path.exists(output_txt):
        return output_txt
    sys.stderr.write(f"[ERR] Expected output missing after ipsae.py: {output_txt}\n")
    return None


def parse_ipsae_txt_for_binder_target(
    txt_path: str,
    binder_chains: Sequence[str],
    target_chains: Sequence[str],
) -> List[float]:
    """
    Extracts ipSAE values from 'asym' rows where Chn1∈binder_chains and Chn2∈target_chains.
    Returns list of ipSAE floats (possibly multiple chain-pairs per model).
    """
    try:
        with open(txt_path, "r") as f:
            lines = [line.rstrip("\n") for line in f]
    except Exception as e:
        sys.stderr.write(f"[ERR] Could not read {txt_path}: {e}\n")
        return []

    header_idx: Dict[str, int] = {}
    values: List[float] = []
    header_found = False

    for line in lines:
        if not line.strip():
            continue
        # Find header
        if not header_found and line.strip().startswith("Chn1 "):
            cols = line.split()
            for i, col in enumerate(cols):
                header_idx[col] = i
            # Required columns
            for req in ("Chn1", "Chn2", "Type", "ipSAE"):
                if req not in header_idx:
                    sys.stderr.write(f"[ERR] Missing column '{req}' in header of {txt_path}\n")
                    return []
            header_found = True
            continue

        if not header_found:
            # Still before header
            continue

        cols = line.split()
        # Guard against malformed lines
        if len(cols) <= max(header_idx.values()):
            continue
        row_type = cols[header_idx["Type"]].strip().lower()
        if row_type != "asym":
            continue
        chn1 = cols[header_idx["Chn1"]].upper()
        chn2 = cols[header_idx["Chn2"]].upper()
        if chn1 in binder_chains and chn2 in target_chains:
            try:
                ipsae_val = float(cols[header_idx["ipSAE"]])
                values.append(ipsae_val)
            except ValueError:
                # Skip non-numeric
                continue
    return values


def reduce_pair_values(values: Sequence[float], mode: str) -> Optional[float]:
    if not values:
        return None
    if mode == "max":
        return max(values)
    if mode == "mean":
        return float(sum(values) / len(values))
    return None


def process_model(task: ModelTask) -> ModelResult:
    txt_path = run_ipsae_if_needed(task)
    if not txt_path:
        return ModelResult(binder_name=task.binder_name, ipsae_value=None, error="ipsae.py failed or output missing")

    values = parse_ipsae_txt_for_binder_target(
        txt_path=txt_path,
        binder_chains=task.binder_chains,
        target_chains=task.target_chains,
    )
    agg = reduce_pair_values(values, mode=task.pair_reduction)
    if agg is None:
        return ModelResult(
            binder_name=task.binder_name,
            ipsae_value=None,
            error="No binder->target asym rows found",
        )
    return ModelResult(binder_name=task.binder_name, ipsae_value=agg)


def aggregate_by_binder(results: Iterable[ModelResult]) -> List[Tuple[str, float, float]]:
    """
    Returns list of (binder_name, mean_ipSAE, stdev_ipSAE).
    Uses sample standard deviation if n>=2 else 0.0.
    """
    by_binder: Dict[str, List[float]] = {}
    for r in results:
        if r.ipsae_value is None:
            continue
        by_binder.setdefault(r.binder_name, []).append(float(r.ipsae_value))

    summary: List[Tuple[str, float, float]] = []
    for binder, vals in by_binder.items():
        avg = float(sum(vals) / len(vals))
        if len(vals) >= 2:
            stdev = float(statistics.stdev(vals))
        else:
            stdev = 0.0
        summary.append((binder, avg, stdev))

    # Sort by descending average ipSAE
    summary.sort(key=lambda t: t[1], reverse=True)
    return summary


def write_csv(rows: List[Tuple[str, float, float]], out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["binder", "average_ipSAE", "stdev_ipSAE"])
        for binder, avg, stdev in rows:
            writer.writerow([binder, f"{avg:.6f}", f"{stdev:.6f}"])


def main() -> None:
    args = parse_args()
    root_dir = os.path.abspath(args.root)
    ipsae_py = os.path.abspath(os.path.expanduser(args.ipsae))
    source = args.source
    pae_cutoff = clamp_pae(args.pae, source)
    dist_cutoff = int(args.dist)
    binder_chains = to_chain_tuple(args.binder_chains)
    target_chains = to_chain_tuple(args.target_chains)
    out_csv = args.out or os.path.join(root_dir, "ipsae_summary.csv")
    pair_reduction = args.pair_reduction

    subdirs = discover_run_subdirs(root_dir)
    if not subdirs:
        raise SystemExit(f"No subdirectories found in {root_dir}")

    tasks: List[ModelTask] = []
    
    if source == "af3":
        # AlphaFold 3 mode
        for run_dir in subdirs:
            binder_name = os.path.basename(run_dir.rstrip(os.sep))
            pairs = find_af3_model_pairs(run_dir)
            if not pairs:
                sys.stderr.write(f"[WARN] No AF3 model pairs found in {run_dir}\n")
                continue
            for json_path, cif_path in pairs:
                tasks.append(
                    ModelTask(
                        binder_name=binder_name,
                        json_path=json_path,
                        cif_path=cif_path,
                        pae_cutoff=pae_cutoff,
                        dist_cutoff=dist_cutoff,
                        ipsae_py=ipsae_py,
                        force=args.force,
                        binder_chains=binder_chains,
                        target_chains=target_chains,
                        pair_reduction=pair_reduction,
                    )
                )
    
    elif source == "boltz":
        # Boltz-2 mode
        prediction_type = args.prediction_type
        for run_dir in subdirs:
            binder_name = os.path.basename(run_dir.rstrip(os.sep))
            
            # Find all prediction directories for this binder
            pred_dirs = find_boltz_prediction_dirs(run_dir, prediction_type)
            if not pred_dirs:
                sys.stderr.write(f"[WARN] No Boltz-2 prediction dirs found in {run_dir}\n")
                continue
            
            for pred_dir in pred_dirs:
                pairs = find_boltz_model_pairs(pred_dir)
                if not pairs:
                    sys.stderr.write(f"[WARN] No Boltz-2 model pairs found in {pred_dir}\n")
                    continue
                
                for pae_path, cif_path in pairs:
                    tasks.append(
                        ModelTask(
                            binder_name=binder_name,
                            json_path=pae_path,  # .npz file for Boltz
                            cif_path=cif_path,
                            pae_cutoff=pae_cutoff,
                            dist_cutoff=dist_cutoff,
                            ipsae_py=ipsae_py,
                            force=args.force,
                            binder_chains=binder_chains,
                            target_chains=target_chains,
                            pair_reduction=pair_reduction,
                        )
                    )

    if not tasks:
        raise SystemExit("No tasks to process.")

    print(f"Found {len(tasks)} model(s) to process from {len(subdirs)} binder(s)...")

    results: List[ModelResult] = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        future_to_task = {pool.submit(process_model, t): t for t in tasks}
        for future in as_completed(future_to_task):
            t = future_to_task[future]
            try:
                res = future.result()
            except Exception as e:
                sys.stderr.write(f"[ERR] Unexpected error for {t.cif_path}: {e}\n")
                res = ModelResult(binder_name=t.binder_name, ipsae_value=None, error=str(e))
            if res.error:
                sys.stderr.write(f"[WARN] {t.binder_name}: {res.error}\n")
            results.append(res)

    summary_rows = aggregate_by_binder(results)
    if not summary_rows:
        raise SystemExit("No successful ipSAE values found to summarize.")

    write_csv(summary_rows, out_csv)
    print(f"Wrote summary CSV: {out_csv}")
    print(f"Summarized {len(summary_rows)} binder(s) with valid ipSAE values.")


if __name__ == "__main__":
    main()

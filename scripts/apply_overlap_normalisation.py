#!/usr/bin/env python3
"""
Apply fitted overlap normalization curves and visualize raw vs normalized distributions.

Generates a 2x5 figure:
  Row 0: Raw per-L overlap values vs normalized exponent (α_new).
  Row 1: Normalized z = (O - μ_L(α_new)) / σ_L(α_new) vs normalized exponent.

Each column corresponds to L = 0..4.

Inputs (from fit_overlap_normalization.py output directory):
  overlap_norm_meta.json  (contains alpha_min / alpha_max)
  overlap_fit_curves.npz  (contains x_grid, mean_grid[5,*], std_grid[5,*])

Example:
  python scripts/apply_overlap_normalization.py \
      --data-dir /path/to/mols \
      --fit-dir plots/overlap_norm_fit \
      --num-mols 200 \
      --outfig overlap_norm_scatter.png
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# tqdm progress (graceful fallback)
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore

def progress(seq, desc: str):
    return tqdm(seq, desc=desc, mininterval=0.5) if tqdm is not None else seq

# Try torch (optional)
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

# Import CustomMolecule (adjust path if needed)
from overlap_pred.custom_data_2 import CustomMolecule  # type: ignore

# L spans consistent with fit_overlap_normalization.py
L_SPANS: List[Tuple[int, int]] = [(0, 1), (1, 4), (4, 9), (9, 16), (16, 25)]


def normalize_exponents(a: np.ndarray, a_min: float, a_max: float) -> np.ndarray:
    """Map positive exponents to [0,1] with log spacing."""
    a = np.asarray(a, dtype=np.float64)
    if not (np.isfinite(a_min) and np.isfinite(a_max) and a_min > 0 and a_max > 0):
        return np.zeros_like(a)
    if math.isclose(a_min, a_max):
        return np.zeros_like(a)
    denom = math.log(a_max / a_min)
    if denom == 0:
        return np.zeros_like(a)
    a_c = np.clip(a, a_min, a_max)
    return np.log(a_c / a_min) / denom


def try_load_molecule(p: Path) -> CustomMolecule | None:
    try:
        m = CustomMolecule.load_pickle(p)
        if m is None:
            return None
        if getattr(m, "exponent_values", None) is None:
            return None
        if getattr(m, "overlap_int_2d", None) is None:
            return None
        return m
    except Exception:
        return None


def extract_perL_values(mol: CustomMolecule) -> tuple[np.ndarray, np.ndarray, int, int]:
    exps = mol.exponent_values
    ovs = mol.overlap_int_2d
    if torch is not None and hasattr(exps, "detach"):
        exps = exps.detach().cpu().numpy()
    if torch is not None and hasattr(ovs, "detach"):
        ovs = ovs.detach().cpu().numpy()
    exps = np.asarray(exps)
    ovs = np.asarray(ovs)
    if exps.ndim != 1 or ovs.ndim != 2 or ovs.shape[0] != exps.shape[0]:
        raise ValueError("Shape mismatch")
    elif getattr(mol, "atom_types", None) is not None:
        at = getattr(mol, "atom_types", None)
        if at is None:
            raise ValueError("Missing atom count")
        if torch is not None and hasattr(at, "detach"):
            at = at.detach().cpu().numpy()
        n_atoms = int(len(at))
    M = ovs.shape[1]
    if M % n_atoms != 0:
        raise ValueError("Overlap columns not divisible by n_atoms")
    pab = M // n_atoms
    if pab < 25:
        raise ValueError("Per-atom basis < 25")
    return exps, ovs, n_atoms, pab


def load_fits(fit_dir: Path):
    meta_path = fit_dir / "overlap_norm_meta.json"
    fit_npz = fit_dir / "overlap_fit_curves.npz"
    if not meta_path.exists() or not fit_npz.exists():
        raise FileNotFoundError(f"Missing required files in {fit_dir}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    a_min = float(meta["alpha_min"])
    a_max = float(meta["alpha_max"])
    npz = np.load(fit_npz)
    xg = npz["x_grid"]
    mean_grid = npz["mean_grid"]  # shape (5, N)
    std_grid = npz["std_grid"]    # shape (5, N)
    f_mean = [PchipInterpolator(xg, mean_grid[L], extrapolate=True) for L in range(5)]
    f_std = [PchipInterpolator(xg, np.maximum(std_grid[L], 1e-12), extrapolate=True) for L in range(5)]
    return a_min, a_max, f_mean, f_std


def main():
    ap = argparse.ArgumentParser(description="Apply overlap normalization fits and plot distributions.")
    ap.add_argument("--data-dir", required=True, type=str, help="Directory containing molecule .pkl files")
    ap.add_argument("--fit-dir", required=True, type=str, help="Directory containing fitted curves (meta JSON + NPZ)")
    ap.add_argument("--num-mols", type=int, default=200, help="Number of molecules to sample (0=all)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    ap.add_argument("--flatten-per-atom", action="store_true", help="Flatten all m-channels instead of per-atom mean over m")
    ap.add_argument("--max-points", type=int, default=200_000_000_000, help="Cap total plotted points per row (subsample if exceeded)")
    ap.add_argument("--outfig", type=str, default="overlap_raw_vs_normalized.png", help="Output figure path")
    ap.add_argument("--report-multiplicity", action="store_true",
                    help="After processing, report expected per-L multiplicity ratios and hypothetical counts if flattened.")
    args = ap.parse_args()
    start_time = time.time()
    data_dir = Path(args.data_dir)
    assert data_dir.is_dir(), f"Data dir not found: {data_dir}"
    fit_dir = Path(args.fit_dir)
    a_min, a_max, f_mean, f_std = load_fits(fit_dir)

    files_all = sorted(data_dir.glob("*.pkl"))
    if not files_all:
        print("No .pkl files found.")
        return 1
    print(f"Found {len(files_all)} .pkl files in {data_dir}")

    files = files_all
    if args.num_mols and args.num_mols > 0 and len(files_all) > args.num_mols:
        rng = random.Random(args.seed)
        files = rng.sample(files_all, args.num_mols)
        print(f"Sampling {len(files)} molecules (requested num-mols={args.num_mols})")
    else:
        print(f"Using all {len(files)} molecules (no sampling)")

    # Accumulate per-L data
    x_per_L_raw = [[] for _ in range(5)]
    y_per_L_raw = [[] for _ in range(5)]
    y_per_L_norm = [[] for _ in range(5)]
    rng_np = np.random.default_rng(args.seed)
    processed = 0
    skipped = 0

    for fp in progress(files, desc="Loading / normalizing"):
        mol = try_load_molecule(fp)
        if mol is None:
            skipped += 1
            continue
        try:
            exps, ovs, n_atoms, pab = extract_perL_values(mol)
        except Exception:
            skipped += 1
            continue
        processed += 1

        # Normalize exponents (vector)
        alpha_new = normalize_exponents(exps, a_min, a_max)
        alpha_new = np.clip(alpha_new, 0.0, 1.0)  # (E,)
        if alpha_new.size == 0:
            continue

        # Evaluate mean/std interpolators in batch => (5, E)
        # PchipInterpolator supports vector inputs.
        mu_mat = np.vstack([fm(alpha_new) for fm in f_mean])          # shape (5, E)
        std_mat = np.vstack([fs(alpha_new) for fs in f_std])          # shape (5, E)
        std_mat = np.clip(std_mat, 1e-12, None)

        # Reshape overlaps to (E, n_atoms, pab); only first 25 channels needed.
        E, M = ovs.shape
        arr = ovs.reshape(E, n_atoms, pab)
        if arr.shape[2] != 25:
            print(f"Skipping {fp}: expected 25 per-atom basis functions, got {arr.shape[2]}")
            skipped += 1
            processed -= 1
            continue

        # For each L span (vectorized over E & atoms)
        for L, (s, t) in enumerate(L_SPANS):
            block = arr[:, :, s:t]  # (E, n_atoms, mL)
            if block.size == 0:
                continue
            if args.flatten_per_atom:
                # Flatten atoms*mL per exponent row
                raw_vals = block.reshape(E, -1)  # (E, n_atoms*mL)
            else:
                # Mean over m channels -> (E, n_atoms)
                raw_vals = block.mean(axis=2)

            # Mask finite raw values
            raw_vals = np.asarray(raw_vals)
            finite_mask = np.isfinite(raw_vals)
            if not finite_mask.any():
                continue

            # Raw values flattened
            raw_flat = raw_vals[finite_mask]

            # Corresponding alpha_new repeated per value
            # Build index of exponents for each finite entry
            # Get row indices from unravel of flattened boolean mask
            exp_indices = np.nonzero(finite_mask)[0]
            alpha_rep = alpha_new[exp_indices]

            # Append raw
            x_per_L_raw[L].append(alpha_rep.astype(np.float64))
            y_per_L_raw[L].append(raw_flat.astype(np.float64))

            # Normalization: broadcast mean/std
            mu_L = mu_mat[L][:, None] if raw_vals.ndim == 2 else mu_mat[L]
            std_L = std_mat[L][:, None] if raw_vals.ndim == 2 else std_mat[L]
            if args.flatten_per_atom:
                mu_L = np.repeat(mu_mat[L], raw_vals.shape[1]).reshape(E, raw_vals.shape[1])
                std_L = np.repeat(std_mat[L], raw_vals.shape[1]).reshape(E, raw_vals.shape[1])

            z_vals = (raw_vals - mu_L) / std_L
            z_vals = z_vals[finite_mask]
            z_vals = z_vals[np.isfinite(z_vals)]
            if z_vals.size:
                y_per_L_norm[L].append(z_vals.astype(np.float64))

    print(f"Finished processing: processed={processed}, skipped={skipped}")
    # Concatenate & subsample
    for L in range(5):
        if x_per_L_raw[L]:
            x_per_L_raw[L] = np.concatenate(x_per_L_raw[L])
            y_per_L_raw[L] = np.concatenate(y_per_L_raw[L])
        else:
            x_per_L_raw[L] = np.array([])
            y_per_L_raw[L] = np.array([])
        if y_per_L_norm[L]:
            y_per_L_norm[L] = np.concatenate(y_per_L_norm[L])
        else:
            y_per_L_norm[L] = np.array([])

        # Subsample (raw & normalized share x array)
        if x_per_L_raw[L].size > args.max_points:
            idx = rng_np.choice(x_per_L_raw[L].size, size=args.max_points, replace=False)
            x_per_L_raw[L] = x_per_L_raw[L][idx]
            y_per_L_raw[L] = y_per_L_raw[L][idx]
            print(f"L={L}: subsampled raw to {args.max_points}")
        if y_per_L_norm[L].size > args.max_points:
            idx = rng_np.choice(y_per_L_norm[L].size, size=args.max_points, replace=False)
            y_per_L_norm[L] = y_per_L_norm[L][idx]
            print(f"L={L}: subsampled norm to {args.max_points}")

    # Per-L summary
    for L in range(5):
        print(f"L={L}: raw_points={x_per_L_raw[L].size}, normalized_points={y_per_L_norm[L].size}")

    # Explain identical counts if averaging over m
    if not args.flatten_per_atom:
        degeneracies = [t - s for (s, t) in L_SPANS]  # [1,3,5,7,9]
        all_equal = len({x_per_L_raw[L].size for L in range(5)}) == 1
        if all_equal:
            print(
                "Info: Identical per-L counts observed because m-components within each L "
                "are being averaged (default behavior). Degeneracies per L: "
                f"{degeneracies}. Use --flatten-per-atom to retain all m values."
            )
        if args.report_multiplicity and x_per_L_raw[0].size > 0:
            base = x_per_L_raw[0].size
            print("Hypothetical raw point counts per L if --flatten-per-atom were used (approx):")
            for L, mult in enumerate(degeneracies):
                approx = base * mult
                print(f"  L={L} (degeneracy={mult}): ~{approx} points")
    else:
        print("Flatten mode: counts should scale with degeneracies (1,3,5,7,9).")

    # Plot
    # REPLACED 2x5 -> 3x5 (added third row with fixed y-limits scatter of normalized values)
    fig, axes = plt.subplots(3, 5, figsize=(5 * 3.2, 3 * 3.0), sharex=True)
    for L in range(5):
        # Row 0: Raw
        ax_raw = axes[0, L]
        xr = x_per_L_raw[L]
        yr = y_per_L_raw[L]
        if xr.size == 0:
            ax_raw.set_title(f"L={L} (no data)")
        else:
            if xr.size > 50000000000:
                hb = ax_raw.hexbin(xr, yr, gridsize=60, cmap="viridis", mincnt=1)
                if L == 4:
                    cbar = fig.colorbar(hb, ax=ax_raw, fraction=0.046, pad=0.04)
                    cbar.set_label("count")
            else:
                ax_raw.scatter(xr, yr, s=0.2, alpha=0.9)
            ax_raw.set_title(f"L={L} raw")
        if L == 0:
            ax_raw.set_ylabel("O (raw)")
        ax_raw.grid(alpha=0.25)

        # Row 1: Normalized (existing behavior: hexbin threshold allowed)
        ax_n = axes[1, L]
        xn = x_per_L_raw[L]
        yn = y_per_L_norm[L]
        if xn.size == 0:
            ax_n.set_title(f"L={L} (no data)")
        else:
            if xn.size > 50000:
                hb2 = ax_n.hexbin(xn, yn, gridsize=60, cmap="plasma", mincnt=1)
                if L == 4:
                    cbar2 = fig.colorbar(hb2, ax=ax_n, fraction=0.046, pad=0.04)
                    cbar2.set_label("count")
            else:
                ax_n.scatter(xn, yn, s=0.2, alpha=0.9, color="tab:orange")
            ax_n.axhline(0, color="k", lw=1)
            ax_n.axhline(3, color="gray", lw=0.7, ls="--")
            ax_n.axhline(-3, color="gray", lw=0.7, ls="--")
            ax_n.set_title(f"L={L} normalized")
        if L == 0:
            ax_n.set_ylabel("z = (O - μ)/σ")
        ax_n.grid(alpha=0.25)

        # Row 2: Normalized scatter (always scatter, fixed y-limits)
        ax_ns = axes[2, L]
        if xn.size == 0:
            ax_ns.set_title(f"L={L} (no data)")
        else:
            # Always scatter regardless of size
            ax_ns.scatter(xn, yn, s=0.2, alpha=0.9, color="tab:orange")
            ax_ns.axhline(0, color="k", lw=1)
            ax_ns.axhline(3, color="gray", lw=0.6, ls="--")
            ax_ns.axhline(-3, color="gray", lw=0.6, ls="--")
            ax_ns.set_title(f"L={L} norm scatter")
        ax_ns.set_ylim(-4.5, 4.5)
        if L == 0:
            ax_ns.set_ylabel("z (±3.5)")
        ax_ns.set_xlabel("α_new")
        ax_ns.grid(alpha=0.25)

    fig.suptitle("Overlap vs normalized exponent (raw / normalized / normalized scatter)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    outfig = Path(args.outfig)
    outfig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfig, dpi=140)
    print(f"Saved figure to {outfig}")
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.2f}s  (avg {elapsed/max(processed,1):.4f}s per molecule)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

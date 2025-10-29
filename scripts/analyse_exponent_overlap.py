#!/usr/bin/env python3
"""
Analyze how normalized exponent values affect overlap integrals.

Loads a sample of CustomMolecule pickles, normalizes exponent values per molecule:
    alpha_new = log(alpha/alpha_min) / log(alpha_max/alpha_min)
and aggregates statistics of overlap integrals per exponent. Saves visualizations.

Expected data layout per molecule:
- exponent_values: shape (n_exponents,) ~ 18
- overlap_int_2d: shape (n_exponents, n_atoms * 25) [sum_{l=0..4} (2l+1) = 25]

Usage:
  python scripts/analyze_exponent_overlap.py \
      --data-dir /export/data/hmichael/scdp/data/full_comp_new \
      --sample-size 20 \
      --outdir plots/exp_overlap_analysis
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import random
import sys
from typing import List, Tuple

import numpy as np
import json
from collections import defaultdict
try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # fallback

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

# Optional interactive plotting (plotly)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _plotly_available = True
except Exception:
    go = None
    make_subplots = None
    _plotly_available = False

# Try optional dependencies for neighbor graph
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore
try:
    from torch_geometric.nn import radius_graph as _radius_graph  # type: ignore
except Exception:
    _radius_graph = None  # type: ignore

# Ensure repo root is on sys.path when running directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from overlap_pred.custom_data_2 import CustomMolecule


def try_load_molecule(pkl_path: Path) -> CustomMolecule | None:
    """Attempt to load a molecule as CustomMolecule, with fallback from compressed.

    Returns None on failure.
    """
    try:
        mol = CustomMolecule.load_pickle(pkl_path)
        # Basic sanity check
        if mol is None or mol.exponent_values is None or mol.overlap_int_2d is None:
            raise ValueError("Loaded object missing exponent_values or overlap_int_2d")
        return mol
    except Exception:
        # Fallback: some datasets may store CompressedCustomMolecule
        try:
            from overlap_pred.compressed_custom_data import CompressedCustomMolecule  # type: ignore
            cm = CompressedCustomMolecule.load_pickle(pkl_path)
            mol = CustomMolecule.from_compressed_custom_molecule(cm)
            if mol is None or mol.exponent_values is None or mol.overlap_int_2d is None:
                return None
            return mol
        except Exception:
            return None


def normalize_exponents(alphas: np.ndarray, global_min: float, global_max: float) -> np.ndarray:
    """Normalize exponents to [0,1] via log spacing with global min/max.

    alpha_new = log(alpha/alpha_min) / log(alpha_max/alpha_min)

    Handles edge cases by returning zeros when invalid.
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    if global_min is None or global_max is None:
        raise ValueError("global_min and global_max must be provided for normalization")
    if not np.isfinite(global_min) or not np.isfinite(global_max) or global_min <= 0 or global_max <= 0:
        return np.zeros_like(alphas)
    if math.isclose(global_min, global_max):
        return np.zeros_like(alphas)
    denom = math.log(global_max / global_min)
    if denom == 0.0:
        return np.zeros_like(alphas)
    # Clip alphas to be at least global_min to avoid negative due to tiny numeric noise
    alphas_clipped = np.clip(alphas, global_min, global_max)
    return np.log(alphas_clipped / global_min) / denom


def compute_global_alpha_minmax(files: List[Path], sample_size: int | None = None) -> tuple[float, float, int]:
    """Compute global min and max of exponent_values across dataset.

    Args:
        files: list of pickle file paths
        sample_size: if provided, randomly sample this many files for estimation; if None, scan all
    Returns:
        (alpha_min, alpha_max, used_file_count)
    """
    if sample_size is not None and sample_size > 0 and len(files) > sample_size:
        rng_local = random.Random(123)
        files_iter = rng_local.sample(files, sample_size)
    else:
        files_iter = files

    global_min = math.inf
    global_max = -math.inf
    used = 0
    for fp in files_iter:
        mol = try_load_molecule(fp)
        if mol is None or mol.exponent_values is None:
            continue
        try:
            exps = mol.exponent_values.detach().cpu().numpy() if hasattr(mol.exponent_values, 'detach') else np.asarray(mol.exponent_values)
        except Exception:
            continue
        if exps.size == 0:
            continue
        exps = exps[np.isfinite(exps)]
        if exps.size == 0:
            continue
        local_min = float(np.min(exps))
        local_max = float(np.max(exps))
        if np.isfinite(local_min) and local_min > 0:
            global_min = min(global_min, local_min)
        if np.isfinite(local_max) and local_max > 0:
            global_max = max(global_max, local_max)
        used += 1
        del mol

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise RuntimeError("Failed to compute global exponent min/max (no valid exponents found)")
    return float(global_min), float(global_max), used


def compute_global_log_alpha_meanstd(files: List[Path], sample_size: int | None = None) -> tuple[float, float, int]:
    """Compute global mean/std of log(exponent_values) across dataset.

    Only positive, finite exponents are used. Returns (mean, std, used_file_count).
    If std <= 0, a fallback std of 1.0 is returned.
    """
    if sample_size is not None and sample_size > 0 and len(files) > sample_size:
        rng_local = random.Random(456)
        files_iter = rng_local.sample(files, sample_size)
    else:
        files_iter = files
    # Welford
    n = 0
    mean = 0.0
    M2 = 0.0
    used = 0
    for fp in files_iter:
        mol = try_load_molecule(fp)
        if mol is None or mol.exponent_values is None:
            continue
        try:
            exps = mol.exponent_values.detach().cpu().numpy() if hasattr(mol.exponent_values, 'detach') else np.asarray(mol.exponent_values)
            if exps.size == 0:
                continue
            exps = exps[np.isfinite(exps) & (exps > 0)]
            if exps.size == 0:
                continue
            logs = np.log(exps.astype(np.float64))
            # Chunk update
            m_chunk = float(logs.mean())
            var_chunk = float(logs.var(ddof=0))
            k = logs.size
            if k == 0:
                continue
            n_new = n + k
            delta = m_chunk - mean
            mean += delta * (k / n_new)
            # Combine variances
            M2 += var_chunk * k + (delta ** 2) * n * k / n_new
            n = n_new
            used += 1
        except Exception:
            continue
    if n < 2:
        return mean, 1.0, used
    var = M2 / (n - 1)
    std = math.sqrt(max(var, 0.0))
    if not (np.isfinite(std) and std > 0):
        std = 1.0
    return mean, std, used


def normalize_exponents_logz(alphas: np.ndarray, log_mean: float, log_std: float) -> np.ndarray:
    """Normalize exponents via z-score in log domain: (log(alpha) - μ_log)/σ_log.

    Non-positive or non-finite alphas yield 0 after replacement. Falls back to zeros if std invalid.
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    if not (np.isfinite(log_mean) and np.isfinite(log_std) and log_std > 0):
        return np.zeros_like(alphas)
    with np.errstate(divide='ignore', invalid='ignore'):
        logs = np.log(np.clip(alphas, 1e-300, None))
        z = (logs - log_mean) / log_std
        z[~np.isfinite(z)] = 0.0
    return z


# ---------------- Per-L overlap statistics (global) -----------------
L_SPANS: List[Tuple[int, int]] = [(0, 1), (1, 4), (4, 9), (9, 16), (16, 25)]  # (start, end) within per-atom block


def compute_global_overlap_L_stats(files: List[Path]) -> tuple[list[float], list[float], list[int]]:
    """Compute global mean and std for overlap values per L channel (L=0..4).

    For each molecule we iterate over atoms; for each atom's per-atom block we
    take the first 25 positions (if available) and aggregate values for each L span.

    Returns:
        (means, stds, counts) each length 5.
    """
    means = [0.0] * 5
    M2 = [0.0] * 5
    counts = [0] * 5

    def _update(L: int, arr: np.ndarray):
        nonlocal means, M2, counts
        if arr.size == 0:
            return
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        n_old = counts[L]
        n_add = arr.size
        n_new = n_old + n_add
        mean_chunk = float(arr.mean())
        # Update mean
        delta = mean_chunk - means[L]
        means[L] += delta * (n_add / n_new)
        # Update M2: sum of squared diffs
        var_chunk = float(arr.var(ddof=0))  # population variance for chunk
        # Parallel variance combination formula
        M2[L] += var_chunk * n_add + (delta ** 2) * n_old * n_add / n_new
        counts[L] = n_new

    for fp in files:
        mol = try_load_molecule(fp)
        if mol is None:
            continue
        try:
            overlaps = mol.overlap_int_2d.detach().cpu().numpy() if hasattr(mol.overlap_int_2d, 'detach') else np.asarray(mol.overlap_int_2d)
            atom_types = None
            if hasattr(mol, 'atom_types') and mol.atom_types is not None:
                try:
                    atom_types = mol.atom_types.detach().cpu().numpy()
                except Exception:
                    atom_types = np.asarray(mol.atom_types)
            if overlaps is None or overlaps.ndim != 2 or atom_types is None:
                continue
            n_atoms = len(atom_types)
            if n_atoms <= 0:
                continue
            cols = overlaps.shape[1]
            if cols % n_atoms != 0:
                continue
            pab = cols // n_atoms
            if pab < 25:
                # Need full L=0..4 coverage
                continue
            for a_idx in range(n_atoms):
                base = a_idx * pab
                for L, (s, e) in enumerate(L_SPANS):
                    if e > pab:
                        continue
                    block = overlaps[:, base + s: base + e]
                    if block.size == 0:
                        continue
                    _update(L, block.ravel())
        except Exception:
            continue

    stds = []
    for L in range(5):
        if counts[L] > 1:
            var = M2[L] / (counts[L] - 1)
            stds.append(math.sqrt(max(var, 0.0)))
        else:
            stds.append(float('nan'))
    return [float(m) for m in means], stds, [int(c) for c in counts]


# ---------------- Curve-based normalization using fitted μ/σ(x) -----------------
def _load_curve_norm_dir(norm_dir: Path):
    """Load fitted normalization curves and meta from a directory.

    Expects files:
      - overlap_norm_meta.json with keys: alpha_min, alpha_max
      - overlap_fit_curves.npz with arrays: x_grid (G,), mean_grid (5,G), std_grid (5,G)

    Returns dict or None on failure.
    """
    try:
        meta_path = norm_dir / 'overlap_norm_meta.json'
        curves_path = norm_dir / 'overlap_fit_curves.npz'
        if not meta_path.exists() or not curves_path.exists():
            return None
        import json as _json
        with open(meta_path, 'r') as f:
            meta = _json.load(f)
        npz = np.load(curves_path)
        x_grid = np.asarray(npz['x_grid'], dtype=np.float64)
        mean_grid = np.asarray(npz['mean_grid'], dtype=np.float64)
        std_grid = np.asarray(npz['std_grid'], dtype=np.float64)
        if x_grid.ndim != 1 or mean_grid.shape != (5, x_grid.size) or std_grid.shape != (5, x_grid.size):
            return None
        a_min = float(meta.get('alpha_min', float('nan')))
        a_max = float(meta.get('alpha_max', float('nan')))
        if not (np.isfinite(a_min) and np.isfinite(a_max) and a_min > 0 and a_max > a_min):
            return None
        return {
            'alpha_min': a_min,
            'alpha_max': a_max,
            'x_grid': x_grid,
            'mean_grid': mean_grid,
            'std_grid': std_grid,
        }
    except Exception:
        return None


def _interp_mu_sigma_for_L(x_vals: np.ndarray, x_grid: np.ndarray, mean_grid: np.ndarray, std_grid: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate μ_L(x) and σ_L(x) at x_vals in [0,1] using numpy.interp.

    Ensures σ>0 by replacing non-positive with nan to be filtered by caller.
    """
    xv = np.asarray(x_vals, dtype=np.float64)
    # Clip to grid bounds to avoid NaNs
    xv = np.clip(xv, float(x_grid[0]), float(x_grid[-1]))
    mu = np.interp(xv, x_grid, mean_grid[L])
    sd = np.interp(xv, x_grid, std_grid[L])
    # Guard against non-positive std
    sd = np.where(sd > 0.0, sd, np.nan)
    return mu, sd


def apply_overlap_L_normalization_inplace(mol: CustomMolecule, mean_L: List[float], std_L: List[float]):
    """In-place (best effort) per-L z-normalization of mol.overlap_int_2d.

    Only normalizes the first 25 positions of each per-atom block corresponding to L=0..4 spans.
    Skips if per-atom basis size < 25 or data invalid.
    """
    try:
        if mol.overlap_int_2d is None:
            return
        is_tensor = hasattr(mol.overlap_int_2d, 'detach')
        arr = mol.overlap_int_2d.detach().cpu().numpy().astype(np.float64) if is_tensor else np.asarray(mol.overlap_int_2d, dtype=np.float64)
        if arr.ndim != 2:
            return
        nE, M = arr.shape
        # Determine n_atoms (need atom_types length if available else infer by divisibility by 25?)
        atom_types = None
        if hasattr(mol, 'atom_types') and mol.atom_types is not None:
            try:
                atom_types = mol.atom_types.detach().cpu().numpy()
            except Exception:
                atom_types = np.asarray(mol.atom_types)
        if atom_types is None:
            return
        n_atoms = len(atom_types)
        if n_atoms <= 0 or M % n_atoms != 0:
            return
        pab = M // n_atoms
        if pab < 25:
            return
        for a_idx in range(n_atoms):
            base = a_idx * pab
            for L, (s, e) in enumerate(L_SPANS):
                if e > pab:
                    continue
                blk = arr[:, base + s: base + e]
                mu = mean_L[L]
                sd = std_L[L]
                if not (np.isfinite(mu) and np.isfinite(sd) and sd > 0):
                    # Replace with zeros relative to mean (centered) if sd invalid
                    blk[:] = 0.0
                else:
                    blk[:] = (blk - mu) / sd
        # Write back
        if is_tensor and torch is not None:
            try:
                mol.overlap_int_2d = torch.as_tensor(arr, dtype=mol.overlap_int_2d.dtype)
            except Exception:
                mol.overlap_int_2d = arr
        else:
            mol.overlap_int_2d = arr
    except Exception:
        return


def aggregate_stats(mol: CustomMolecule, global_min: float, global_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute per-exponent statistics for a single molecule.

    Returns:
        norm_alpha: (E,)
        mean_abs: (E,)
        max_abs: (E,)
        std_abs: (E,)
        meta: dict with molecule-level metadata
    """
    # Convert tensors to numpy
    exps = mol.exponent_values.detach().cpu().numpy() if hasattr(mol.exponent_values, 'detach') else np.asarray(mol.exponent_values)
    overlaps = mol.overlap_int_2d.detach().cpu().numpy() if hasattr(mol.overlap_int_2d, 'detach') else np.asarray(mol.overlap_int_2d)

    if overlaps.ndim != 2 or exps.ndim != 1 or overlaps.shape[0] != exps.shape[0]:
        raise ValueError("Invalid shapes: exps (E,), overlaps (E, M)")

    norm_alpha = normalize_exponents(exps, global_min=global_min, global_max=global_max)
    abs_over = np.abs(overlaps)
    mean_abs = abs_over.mean(axis=1)
    max_abs = abs_over.max(axis=1)
    std_abs = abs_over.std(axis=1)

    n_atoms = int(mol.n_atom) if mol.n_atom is not None else (len(mol.atom_types) if mol.atom_types is not None else -1)
    per_atom_basis = overlaps.shape[1] // n_atoms if n_atoms and n_atoms > 0 else None

    meta = {
        "n_atoms": n_atoms,
        "per_atom_basis": per_atom_basis,
        "mol_id": mol.id if not isinstance(mol.id, (list, tuple)) else (mol.id[0] if mol.id else None),
        "E": overlaps.shape[0],
        "M": overlaps.shape[1],
    }
    return norm_alpha, mean_abs, max_abs, std_abs, meta


def plot_scatter(norm_alphas: np.ndarray, values: np.ndarray, out_path: Path, ylabel: str):
    plt.figure(figsize=(7, 5))
    plt.scatter(norm_alphas, values, s=14, alpha=0.6, edgecolor='none')
    plt.xlabel("Normalized exponent α_new")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_hexbin(norm_alphas: np.ndarray, values: np.ndarray, out_path: Path, ylabel: str):
    plt.figure(figsize=(7, 5))
    hb = plt.hexbin(norm_alphas, values, gridsize=40, cmap='viridis', mincnt=1)
    plt.xlabel("Normalized exponent α_new")
    plt.ylabel(ylabel)
    cb = plt.colorbar(hb)
    cb.set_label('Count')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_binned(norm_alphas: np.ndarray, values: np.ndarray, out_path: Path, ylabel: str, bins: int = 18):
    # Bin by normalized alphas
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(norm_alphas, edges) - 1
    means = np.full(bins, np.nan)
    medians = np.full(bins, np.nan)
    counts = np.zeros(bins, dtype=int)
    for b in range(bins):
        mask = idx == b
        if np.any(mask):
            vv = values[mask]
            means[b] = np.nanmean(vv)
            medians[b] = np.nanmedian(vv)
            counts[b] = int(np.sum(mask))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(centers, means, marker='o', label='Mean')
    ax1.plot(centers, medians, marker='s', label='Median')
    ax1.set_xlabel("Normalized exponent α_new (binned)")
    ax1.set_ylabel(ylabel)
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.bar(centers, counts, width=1.0/bins * 0.9, color='gray', alpha=0.25, label='Count')
    ax2.set_ylabel('Count per bin')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _init_elem_aggr(pos_dim: int, bins: int):
    return {
        'sum': np.zeros((bins, pos_dim), dtype=np.float64),
        'count': np.zeros((bins, pos_dim), dtype=np.float64),
        'pos_dim': pos_dim,
        'bins': bins,
    }


def _ensure_pos_dim(aggr: dict, pos_dim: int):
    """Ensure aggregator pos_dim is <= new pos_dim, truncating if needed to the min.
    Returns the effective pos_dim to use for accumulation (min(current, new)).
    """
    if aggr['pos_dim'] == pos_dim:
        return pos_dim
    new_dim = min(aggr['pos_dim'], pos_dim)
    if new_dim != aggr['pos_dim']:
        # truncate existing arrays
        aggr['sum'] = aggr['sum'][:, :new_dim]
        aggr['count'] = aggr['count'][:, :new_dim]
        aggr['pos_dim'] = new_dim
    return new_dim


def plot_element_heatmaps(element_aggr: dict, edges: np.ndarray, outdir: Path):
    centers = 0.5 * (edges[:-1] + edges[1:])
    for z, aggr in element_aggr.items():
        sums = aggr['sum']
        counts = aggr['count']
        with np.errstate(divide='ignore', invalid='ignore'):
            means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        plt.figure(figsize=(9, 5))
        im = plt.imshow(
            means,
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            cmap='viridis'
        )
        plt.colorbar(im, label='Mean |overlap|')
        plt.xlabel('Per-atom position index')
        plt.ylabel('Normalized exponent bin')
        yticks = np.linspace(0, means.shape[0]-1, min(10, means.shape[0])).astype(int)
        plt.yticks(yticks, [f"{centers[i]:.2f}" for i in yticks])
        plt.title(f'Element Z={z} — mean |overlap| vs normalized exponent and position')
        out_path = outdir / f'heatmap_element_Z{int(z)}.png'
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze normalized exponents vs overlap integrals (global normalization across dataset)")
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with molecule .pkl files')
    parser.add_argument('--sample-size', type=int, default=20, help='Number of molecules to sample')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--outdir', type=str, default='plots/exp_overlap_analysis_2', help='Output directory for plots')
    parser.add_argument('--minmax-scan-sample', type=int, default=0, help='If >0, estimate global min/max from a random subset of this many files; 0 means scan all')
    parser.add_argument('--recompute-minmax', action='store_true', help='Force recompute global min/max even if cached JSON exists in outdir')
    parser.add_argument('--heatmap-bins', type=int, default=18, help='Number of bins along normalized exponent axis for heatmaps')
    parser.add_argument('--heatmap-max-elements', type=int, default=8, help='Max number of elements to plot (top by observations)')
    parser.add_argument('--heatmap-pos', type=int, default=25, help='Max per-atom position dimension to aggregate (truncate if larger)')
    parser.add_argument('--heatmap-elements', type=str, default='', help='Comma-separated atomic numbers to force include (e.g., "1,6,7,8")')
    # Distribution plots
    parser.add_argument('--dist-max-samples-per-index', type=int, default=10000, help='Reservoir sample size per element/index for distribution plots')
    parser.add_argument('--dist-abs', action='store_true', help='Use absolute overlaps for distribution plots')
    # Outlier scan and histogram
    parser.add_argument('--outlier-scan', action='store_true', help='Enable outlier scan based on global mean/std over selected files')
    parser.add_argument('--outlier-abs', action='store_true', help='Use absolute overlaps for outlier scan thresholds')
    parser.add_argument('--outlier-z', type=float, default=3.0, help='Z-score threshold for outliers')
    parser.add_argument('--outlier-max-save', type=int, default=20000, help='Max outlier records to save to JSON (top by |z|)')
    parser.add_argument('--hist-bins', type=int, default=200, help='Number of bins for overlap histogram')
    parser.add_argument('--hist-k-sigma', type=float, default=6.0, help='Histogram range = mean ± k*sigma')
    parser.add_argument('--reuse-outliers-cache', action='store_true', help='Reuse outlier JSON if present to skip re-scan')
    # Per-L scatter plot controls
    parser.add_argument('--lscatter-cap-per-series', type=int, default=10000, help='Max sampled points per (L, element) series for L-scatter plots')
    parser.add_argument('--lscatter-max-elements', type=int, default=12, help='Max elements to show in legend/colors for L-scatter plots (top by sampled points)')
    parser.add_argument('--lscatter-max-neigh-cats', type=int, default=10, help='Max neighbor-type categories to color distinctly (others grouped)')
    parser.add_argument('--lscatter-weight-by-freq', action='store_true', help='Downweight point alpha by neighbor-combination frequency in per-L plots')
    parser.add_argument('--lscatter-stats-lines', action='store_true', help='Overlay binned mean and ±std lines on per-L scatter (per element)')
    parser.add_argument('--lscatter-stats-bins', type=int, default=1000, help='Number of bins along normalized exponent axis for stats line (default 1000)')
    parser.add_argument('--lscatter-local-z', action='store_true', help='Produce additional plots with point-wise z=(O-mu_bin)/std_bin using bin-wise local statistics per (L,Z)')
    parser.add_argument('--lscatter-local-z-bins', type=int, default=400, help='Number of exponent bins for local z-score normalization (default 400)')
    parser.add_argument('--lscatter-aggregate', action='store_true', help='Generate L-only aggregated plots (all atom types merged) with mean/std overlays and normalized z version')
    parser.add_argument('--lscatter-aggregate-bins', type=int, default=800, help='Number of exponent bins for aggregated L plots (default 800)')
    parser.add_argument('--lscatter-aggregate-cap', type=int, default=500000, help='Max points per L after merge before random downsampling (default 500k)')
    parser.add_argument('--interactive-l', type=int, choices=[0,1,2,3,4], default=None, help='If set, only include this L channel (0-4) in the interactive Plotly output')
    # Exponent histogram
    parser.add_argument('--exp-hist-bins', type=int, default=100, help='Number of bins for raw exponent and log-exponent histograms')
    # Neighbor graph (radius graph) params
    parser.add_argument('--neighbor-radius', type=float, default=3.0, help='Radius (in coordinate units) for torch_geometric.radius_graph')
    parser.add_argument('--neighbor-max-num', type=int, default=8, help='Max neighbors per node for torch_geometric.radius_graph')
    # Overlap per-L normalization
    parser.add_argument('--apply-overlap-normalization', action='store_true', help='Apply global per-L (O - mu_L)/sigma_L normalization to overlaps before analysis plots')
    parser.add_argument('--recompute-overlap-stats', action='store_true', help='Force recompute global per-L overlap mean/std even if cached')
    # Curve-based normalization using fitted μ/σ(x)
    parser.add_argument('--curve-norm-dir', type=str, default='', help='Directory with fitted normalization curves (overlap_norm_meta.json, overlap_fit_curves.npz)')
    parser.add_argument('--curve-norm-plots', action='store_true', help='Generate aggregated L plots using curve-based normalization z=(O-μ_L(x))/σ_L(x)')
    parser.add_argument('--curve-norm-cap-per-L', type=int, default=500000, help='Max samples per L for curve-normalized plots (reservoir sampling)')
    parser.add_argument('--curve-norm-bins', type=int, default=400, help='Bins along normalized exponent x for stats overlay in curve-normalized plots')
    parser.add_argument('--curve-norm-ymax', type=float, default=5.0, help='Clamp |z| to this value in curve-normalized plots')
    parser.add_argument('--curve-norm-hist-bins', type=int, default=200, help='Number of bins for curve-normalized per-L z histograms')
    parser.add_argument('--curve-use-global-minmax', action='store_true', help='Use dataset global alpha_min/alpha_max instead of curve meta for x normalization (diagnostic)')
    parser.add_argument('--curve-jitter', type=float, default=0.0, help='Optional small jitter (e.g. 0.002) added to x in curve-normalized scatter to visualize density when exponents are discrete')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    assert data_dir.is_dir(), f"Data dir not found: {data_dir}"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Collect candidate files (all for min/max scan and sampling)
    files_all = sorted(list(data_dir.glob('*.pkl')))
    if not files_all:
        print(f"No .pkl files found in {data_dir}")
        return 1

    # Global cache path (extended to also hold overlap stats)
    cache_path = outdir / 'global_exponent_minmax.json'
    global_min = None
    global_max = None
    used_count = 0
    overlap_mean_L: List[float] | None = None
    overlap_std_L: List[float] | None = None
    overlap_count_L: List[int] | None = None
    log_alpha_mean: float | None = None
    log_alpha_std: float | None = None
    # Curve normalization payload
    curve_payload = None
    if args.curve_norm_plots and args.curve_norm_dir:
        curve_payload = _load_curve_norm_dir(Path(args.curve_norm_dir))
        print("Loaded curve normalization data from", args.curve_norm_dir if curve_payload else "None")
        if curve_payload is None:
            print(f"[WARN] Failed to load curve normalization data from {args.curve_norm_dir}; curve-norm plots will be skipped.")
            args.curve_norm_plots = False

    # Try to reuse cache unless forced to recompute
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as cf:
                cache = json.load(cf)
            if not args.recompute_minmax:
                if 'alpha_min' in cache and 'alpha_max' in cache:
                    global_min = float(cache.get('alpha_min'))
                    global_max = float(cache.get('alpha_max'))
                    used_count = int(cache.get('file_count', 0))
                    print(f"Loaded cached global min/max: min={global_min:.6g}, max={global_max:.6g} (files: {used_count})")
                if 'log_alpha_mean' in cache and 'log_alpha_std' in cache:
                    log_alpha_mean = float(cache.get('log_alpha_mean'))
                    log_alpha_std = float(cache.get('log_alpha_std'))
                    if np.isfinite(log_alpha_mean) and np.isfinite(log_alpha_std):
                        print(f"Loaded cached log-exponent stats: mean={log_alpha_mean:.6g}, std={log_alpha_std:.6g}")
            # Load overlap stats if present and not forcing recompute
            if not args.recompute_overlap_stats and all(k in cache for k in ['overlap_mean_L', 'overlap_std_L', 'overlap_L_count']):
                try:
                    overlap_mean_L = [float(x) for x in cache.get('overlap_mean_L', [])]
                    overlap_std_L = [float(x) for x in cache.get('overlap_std_L', [])]
                    overlap_count_L = [int(x) for x in cache.get('overlap_L_count', [])]
                    if len(overlap_mean_L) == 5 and len(overlap_std_L) == 5:
                        print("Loaded cached per-L overlap stats:")
                        for L in range(5):
                            print(f"  L={L}: mean={overlap_mean_L[L]:.6g}, std={overlap_std_L[L]:.6g}, count={overlap_count_L[L] if overlap_count_L else 'n/a'}")
                except Exception:
                    overlap_mean_L = None
                    overlap_std_L = None
        except Exception:
            global_min = None
            global_max = None

    if global_min is None or global_max is None:
        scan_sample = args.minmax_scan_sample if args.minmax_scan_sample > 0 else None
        print("Computing global exponent min/max across dataset..." + (f" (sample={scan_sample})" if scan_sample else ""))
        global_min, global_max, used_count = compute_global_alpha_minmax(files_all, sample_size=scan_sample)
        print(f"Global exponent min/max: min={global_min:.6g}, max={global_max:.6g} (from {used_count} files)")
        # Cache results
        # Write partial cache (may extend later with overlap stats)
        try:
            existing = {}
            if cache_path.exists():
                try:
                    with open(cache_path, 'r') as cf:
                        existing = json.load(cf)
                except Exception:
                    existing = {}
            existing.update({"alpha_min": global_min, "alpha_max": global_max, "file_count": used_count})
            with open(cache_path, 'w') as cf:
                json.dump(existing, cf)
        except Exception:
            pass

    # Compute global log-exponent mean/std if not cached
    if log_alpha_mean is None or log_alpha_std is None or args.recompute_minmax:
        scan_sample = args.minmax_scan_sample if args.minmax_scan_sample > 0 else None
        print("Computing global log-exponent mean/std across dataset..." + (f" (sample={scan_sample})" if scan_sample else ""))
        log_alpha_mean, log_alpha_std, _ = compute_global_log_alpha_meanstd(files_all, sample_size=scan_sample)
        print(f"Global log-exponent stats: mean={log_alpha_mean:.6g}, std={log_alpha_std:.6g}")
        try:
            existing = {}
            if cache_path.exists():
                try:
                    with open(cache_path, 'r') as cf:
                        existing = json.load(cf)
                except Exception:
                    existing = {}
            existing.update({
                "alpha_min": global_min,
                "alpha_max": global_max,
                "file_count": used_count,
                "log_alpha_mean": log_alpha_mean,
                "log_alpha_std": log_alpha_std,
            })
            with open(cache_path, 'w') as cf:
                json.dump(existing, cf)
        except Exception:
            pass

    # Compute per-L overlap stats if needed for normalization
    if args.apply_overlap_normalization:
        need_overlap_stats = (
            overlap_mean_L is None or overlap_std_L is None or
            len(overlap_mean_L) != 5 or len(overlap_std_L) != 5 or args.recompute_overlap_stats
        )
        if need_overlap_stats:
            print("Computing global per-L overlap mean/std across dataset (may take time)...")
            overlap_mean_L, overlap_std_L, overlap_count_L = compute_global_overlap_L_stats(files_all)
            for L in range(5):
                print(f"  L={L}: mean={overlap_mean_L[L]:.6g}, std={overlap_std_L[L]:.6g}, count={overlap_count_L[L]}")
            # Update cache
            try:
                existing = {}
                if cache_path.exists():
                    try:
                        with open(cache_path, 'r') as cf:
                            existing = json.load(cf)
                    except Exception:
                        existing = {}
                existing.update({
                    "alpha_min": global_min,
                    "alpha_max": global_max,
                    "file_count": used_count,
                    "overlap_mean_L": overlap_mean_L,
                    "overlap_std_L": overlap_std_L,
                    "overlap_L_count": overlap_count_L,
                })
                with open(cache_path, 'w') as cf:
                    json.dump(existing, cf)
            except Exception:
                pass
        else:
            print("Using cached per-L overlap mean/std for normalization.")

    rng = random.Random(args.seed)
    if len(files_all) > args.sample_size:
        files = rng.sample(files_all, args.sample_size)
    else:
        files = files_all[:args.sample_size]

    print(f"Loading {len(files)} molecules from {data_dir} ...")

    all_norm_alpha: List[float] = []
    all_norm_alpha_logz: List[float] = []
    all_mean_abs: List[float] = []
    all_max_abs: List[float] = []
    all_std_abs: List[float] = []
    # Per-exponent per-L mean overlaps (E,5) per molecule -> concatenated later
    all_L_mean_per_exp: List[np.ndarray] = []
    # Molecule id per exponent row (strings), repeated per exponent
    all_mol_ids: List[np.ndarray] = []
    # Interactive aggregator: collect individual overlap points per L (reservoir sampled)
    # Also store a compact atom-types string per point for hover metadata
    interactive_aggr = {L: {'x': [], 'y': [], 'mol_id': [], 'atom_types': [], 'count': 0, 'idx': []} for L in range(5)}
    # Exponent accumulation for histogram
    all_exponents: List[np.ndarray] = []

    # Heatmap aggregation structures
    heat_bins = int(args.heatmap_bins)
    edges = np.linspace(0.0, 1.0, heat_bins + 1)
    element_aggr: dict = {}
    element_aggr_logz: dict = {}
    element_obs_counts = defaultdict(int)  # count of atom-exponent rows contributed
    # Distribution aggregation structures (per element per position reservoir)
    element_dist: dict = {}
    include_elements = set()
    if args.heatmap_elements:
        try:
            include_elements = {int(s) for s in args.heatmap_elements.split(',') if s.strip()}
        except Exception:
            include_elements = set()
    # Track total atom occurrences per element across processed molecules (for histogram normalization)
    element_atom_occurrences = defaultdict(int)
    # Track raw overlap global min/max for per-L histograms
    lhist_vmin = math.inf
    lhist_vmax = -math.inf
    # Accumulator for per-L scatter: dict[L][Z] -> {'x': list, 'y': list, 'count': int}
    lscatter = {L: {} for L in range(5)}
    lscatter_logz = {L: {} for L in range(5)}
    # Aggregators for curve-normalized z vs x per L
    curve_aggr = {L: {'x': [], 'z': [], 'count': 0} for L in range(5)}
    curve_cap = int(args.curve_norm_cap_per_L)

    def _reservoir_add(L: int, x_arr: np.ndarray, z_arr: np.ndarray):
        if not args.curve_norm_plots or curve_payload is None:
            return
        if x_arr.size == 0:
            return
        ag = curve_aggr[L]
        cnt = ag['count']
        # Ensure python lists for storage
        xs = ag['x']
        zs = ag['z']
        for xi, zi in zip(x_arr.tolist(), z_arr.tolist()):
            cnt += 1
            if len(xs) < curve_cap:
                xs.append(xi)
                zs.append(zi)
            else:
                j = rng.randrange(cnt)
                if j < curve_cap:
                    xs[j] = xi
                    zs[j] = zi
        ag['count'] = cnt

    meta_summary = {
        'loaded': 0,
        'skipped': 0,
        'per_atom_basis_values': [],
        'n_atoms_values': [],
        'E_values': [],
    }

    # Helper progress iterator
    def progress(seq, desc):
        if tqdm is None:
            return seq
        total = len(seq) if hasattr(seq, '__len__') else None
        return tqdm(seq, total=total, desc=desc)

    for fp in progress(files, desc='Processing molecules'):
        mol = try_load_molecule(fp)
        if mol is None:
            meta_summary['skipped'] += 1
            print(f"  - Skipped: {fp.name}")
            continue
        # Apply per-L overlap normalization if requested and stats available
        if args.apply_overlap_normalization and overlap_mean_L is not None and overlap_std_L is not None:
            try:
                apply_overlap_L_normalization_inplace(mol, overlap_mean_L, overlap_std_L)
            except Exception:
                pass
        try:
            norm_alpha, mean_abs, max_abs, std_abs, meta = aggregate_stats(mol, global_min=global_min, global_max=global_max)
            # Second normalization (log z)
            try:
                norm_alpha_logz = normalize_exponents_logz(mol.exponent_values.detach().cpu().numpy() if hasattr(mol.exponent_values, 'detach') else np.asarray(mol.exponent_values), log_alpha_mean, log_alpha_std)  # type: ignore[arg-type]
            except Exception:
                norm_alpha_logz = np.zeros_like(norm_alpha)
        except Exception as e:
            meta_summary['skipped'] += 1
            print(f"  - Error processing {fp.name}: {e}")
            continue

        # Basic integrity checks
        if np.any(~np.isfinite(norm_alpha)) or np.any(~np.isfinite(mean_abs)):
            meta_summary['skipped'] += 1
            print(f"  - Non-finite values in {fp.name}, skipping")
            continue

        meta_summary['loaded'] += 1
        if meta.get('per_atom_basis') is not None:
            meta_summary['per_atom_basis_values'].append(int(meta['per_atom_basis']))
        if meta.get('n_atoms') is not None and meta['n_atoms'] > 0:
            meta_summary['n_atoms_values'].append(int(meta['n_atoms']))
        if meta.get('E') is not None:
            meta_summary['E_values'].append(int(meta['E']))

        all_norm_alpha.append(norm_alpha)
        all_norm_alpha_logz.append(norm_alpha_logz)
        all_mean_abs.append(mean_abs)
        all_max_abs.append(max_abs)
        all_std_abs.append(std_abs)
        # Compute per-exponent mean overlap per L (L=0..4) across atoms for this molecule
        try:
            # overlaps and exps should already have been loaded earlier in this block
            overlaps_local = mol.overlap_int_2d.detach().cpu().numpy() if hasattr(mol.overlap_int_2d, 'detach') else np.asarray(mol.overlap_int_2d)
            exps_local = mol.exponent_values.detach().cpu().numpy() if hasattr(mol.exponent_values, 'detach') else np.asarray(mol.exponent_values)
            E = exps_local.shape[0]
            # default: NaNs if we can't compute per-L
            perL = np.full((E, 5), np.nan, dtype=np.float64)
            # require atom split
            if overlaps_local.ndim == 2:
                cols_local = overlaps_local.shape[1]
                # try infer n_atoms from meta if present
                n_atoms_local = int(meta.get('n_atoms', -1)) if meta.get('n_atoms', None) is not None else -1
                if n_atoms_local and n_atoms_local > 0 and cols_local % n_atoms_local == 0:
                    pab_local = cols_local // n_atoms_local
                    # only compute if per-atom block covers L spans
                    if pab_local >= 25:
                        for L, (s, e) in enumerate(L_SPANS):
                            # gather blocks for all atoms for this L span
                            blocks = []
                            for a_idx_local in range(n_atoms_local):
                                base = a_idx_local * pab_local
                                if e > pab_local:
                                    continue
                                blk = overlaps_local[:, base + s: base + e]
                                if blk.size == 0:
                                    continue
                                blocks.append(np.abs(blk))
                            if blocks:
                                # concatenate along columns and compute mean per exponent row
                                cat = np.concatenate(blocks, axis=1)
                                with np.errstate(invalid='ignore'):
                                    perL[:, L] = np.nanmean(cat, axis=1)
            all_L_mean_per_exp.append(perL)
            # Molecule id repeated per exponent for hover metadata
            mid = meta.get('mol_id', None)
            mid_str = str(mid) if mid is not None else ''
            all_mol_ids.append(np.array([mid_str] * (perL.shape[0]), dtype=object))
        except Exception:
            # best-effort; don't fail the main loop on interactive collection errors
            all_L_mean_per_exp.append(np.full((norm_alpha.shape[0], 5), np.nan, dtype=np.float64))
            all_mol_ids.append(np.array([''] * norm_alpha.shape[0], dtype=object))
        # Accumulate raw exponent values for histogram
        try:
            exps_raw = mol.exponent_values.detach().cpu().numpy() if hasattr(mol.exponent_values, 'detach') else np.asarray(mol.exponent_values)
            if exps_raw.size > 0:
                finite_exps = exps_raw[np.isfinite(exps_raw)]
                if finite_exps.size > 0:
                    all_exponents.append(finite_exps.astype(float))
        except Exception:
            pass

        # Track raw overlap min/max for per-L hist range
        try:
            overlaps_nm = mol.overlap_int_2d.detach().cpu().numpy() if hasattr(mol.overlap_int_2d, 'detach') else np.asarray(mol.overlap_int_2d)
            if overlaps_nm.size > 0:
                finite_vals = overlaps_nm[np.isfinite(overlaps_nm)]
                if finite_vals.size > 0:
                    lhist_vmin = min(lhist_vmin, float(np.min(finite_vals)))
                    lhist_vmax = max(lhist_vmax, float(np.max(finite_vals)))
        except Exception:
            pass

        # Heatmap aggregation per element
        try:
            # Prepare arrays for per-atom processing
            exps = mol.exponent_values.detach().cpu().numpy() if hasattr(mol.exponent_values, 'detach') else np.asarray(mol.exponent_values)
            overlaps = mol.overlap_int_2d.detach().cpu().numpy() if hasattr(mol.overlap_int_2d, 'detach') else np.asarray(mol.overlap_int_2d)
            atom_types = mol.atom_types.detach().cpu().numpy().astype(int) if hasattr(mol.atom_types, 'detach') else np.asarray(mol.atom_types).astype(int)
            # Compact string of unique atom types for hover metadata (e.g. "1,6,8")
            try:
                if atom_types is not None and getattr(atom_types, 'size', 0) > 0:
                    uniq = np.unique(atom_types.astype(int))
                    atom_types_str = ','.join(str(int(x)) for x in uniq.tolist())
                else:
                    atom_types_str = ''
            except Exception:
                atom_types_str = ''
            # One-time debug: show raw exponent variety and two normalization modes
            if meta_summary['loaded'] == 1:  # after first successful molecule earlier
                try:
                    raw_exps_dbg = np.asarray(exps, dtype=float)
                    if raw_exps_dbg.size > 0:
                        raw_sort = np.unique(np.sort(raw_exps_dbg))
                        sample = raw_sort[:10]
                        print(f"[debug exp] first molecule unique exponent count={raw_sort.size} sample={sample.tolist()}")
                        if global_min is not None and global_max is not None:
                            x_global_dbg = normalize_exponents(raw_exps_dbg, global_min, global_max)
                            print(f"[debug exp] global norm unique x count={np.unique(np.round(x_global_dbg,6)).size}")
                        if curve_payload is not None:
                            a_min_curve_dbg = float(curve_payload['alpha_min']); a_max_curve_dbg = float(curve_payload['alpha_max'])
                            x_curve_dbg = normalize_exponents(raw_exps_dbg, a_min_curve_dbg, a_max_curve_dbg)
                            print(f"[debug exp] curve meta norm unique x count={np.unique(np.round(x_curve_dbg,6)).size}")
                except Exception:
                    pass
            # Curve-based normalization accumulation (per-L) for this molecule
            if args.curve_norm_plots and curve_payload is not None and (not args.curve_use_global_minmax or (global_min is not None and global_max is not None)):
                try:
                    # Decide which min/max to use for x normalization
                    a_min_curve = float(curve_payload['alpha_min'])
                    a_max_curve = float(curve_payload['alpha_max'])
                    if args.curve_use_global_minmax and global_min is not None and global_max is not None:
                        a_min = float(global_min)
                        a_max = float(global_max)
                    else:
                        a_min = a_min_curve
                        a_max = a_max_curve
                    x_grid = curve_payload['x_grid']
                    mean_grid = curve_payload['mean_grid']
                    std_grid = curve_payload['std_grid']
                    if exps is not None and overlaps is not None and exps.ndim == 1 and overlaps.ndim == 2 and overlaps.shape[0] == exps.shape[0]:
                        # Normalize exponents to x in [0,1] using curve meta
                        x_vals = normalize_exponents(exps, global_min=a_min, global_max=a_max)
                        # Determine per-atom splitting
                        if atom_types is not None and atom_types.size > 0:
                            n_atoms = int(atom_types.shape[0])
                        else:
                            n_atoms = 0
                        M = overlaps.shape[1]
                        if n_atoms > 0 and M % n_atoms == 0:
                            pab = M // n_atoms
                            if pab >= 25:
                                # Precompute μ/σ per L at this molecule's x values
                                muL = {}
                                sdL = {}
                                for L in range(5):
                                    muL[L], sdL[L] = _interp_mu_sigma_for_L(x_vals, x_grid, mean_grid, std_grid, L)
                                # Iterate atoms and L spans, reservoir sample
                                for a_idx in range(n_atoms):
                                    base = a_idx * pab
                                    for L, (s, e) in enumerate(L_SPANS):
                                        if e > pab:
                                            continue
                                        mu = muL[L]
                                        sd = sdL[L]
                                        # Skip if sd is degenerate across all exponents
                                        if not np.any(np.isfinite(sd)):
                                            continue
                                        # Accumulate across positions s..e-1
                                        for j in range(s, e):
                                            y = overlaps[:, base + j].astype(np.float64)
                                            z = (y - mu) / sd
                                            mask = np.isfinite(z) & np.isfinite(x_vals)
                                            if np.any(mask):
                                                _reservoir_add(L, x_vals[mask], z[mask])
                except Exception:
                    pass
            # Update total atom occurrences by element Z
            if atom_types is not None and atom_types.size > 0:
                uniq_z, cnts = np.unique(atom_types, return_counts=True)
                for z_val, c_val in zip(uniq_z.tolist(), cnts.tolist()):
                    element_atom_occurrences[int(z_val)] += int(c_val)
            n_atoms = len(atom_types)
            if n_atoms <= 0:
                continue
            # Build neighbor-type labels per atom using radius graph (PyG) if possible
            neigh_labels = ['none'] * n_atoms
            try:
                neigh_lists = [set() for _ in range(n_atoms)]
                built_graph = False
                # Allowed neighbor atomic numbers (exclude virtual nodes Z=0)
                allowed_neigh = {1, 6, 7, 8}
                # Obtain coordinates
                pos_arr = None
                for attr in ('pos', 'positions', 'coords', 'coordinates', 'xyz', 'R'):
                    if hasattr(mol, attr) and getattr(mol, attr) is not None:
                        arr = getattr(mol, attr)
                        try:
                            pos_arr = arr.detach().cpu().numpy() if hasattr(arr, 'detach') else np.asarray(arr)
                        except Exception:
                            pos_arr = np.asarray(arr)
                        break
                if pos_arr is not None:
                    pos_arr = np.asarray(pos_arr, dtype=np.float32)
                    if pos_arr.ndim == 2 and pos_arr.shape[0] == n_atoms:
                        if _radius_graph is not None and torch is not None:
                            try:
                                pos_t = torch.as_tensor(pos_arr, dtype=torch.float32)
                                ei_t = _radius_graph(pos_t, r=float(args.neighbor_radius), loop=False, max_num_neighbors=int(args.neighbor_max_num))
                                ei = ei_t.detach().cpu().numpy().astype(int)
                                if ei.ndim == 2 and ei.shape[0] == 2:
                                    us = ei[0].ravel()
                                    vs = ei[1].ravel()
                                    for u, v in zip(us, vs):
                                        if 0 <= u < n_atoms and 0 <= v < n_atoms and u != v:
                                            if int(atom_types[v]) in allowed_neigh:
                                                neigh_lists[u].add(int(v))
                                            if int(atom_types[u]) in allowed_neigh:
                                                neigh_lists[v].add(int(u))
                                    built_graph = True
                            except Exception:
                                built_graph = False
                # Fall back to existing adjacency-like attributes if radius graph not built
                if not built_graph:
                    # PyG-style edge_index (2, E)
                    if hasattr(mol, 'edge_index') and mol.edge_index is not None:
                        ei = mol.edge_index.detach().cpu().numpy() if hasattr(mol.edge_index, 'detach') else np.asarray(mol.edge_index)
                        if ei.ndim == 2 and ei.shape[0] == 2:
                            us = ei[0].astype(int).ravel()
                            vs = ei[1].astype(int).ravel()
                            for u, v in zip(us, vs):
                                if 0 <= u < n_atoms and 0 <= v < n_atoms:
                                    if int(atom_types[v]) in allowed_neigh:
                                        neigh_lists[u].add(int(v))
                                    if int(atom_types[u]) in allowed_neigh:
                                        neigh_lists[v].add(int(u))
                    # bond_index or bonds as (E, 2)
                    elif hasattr(mol, 'bond_index') and mol.bond_index is not None:
                        bi = mol.bond_index.detach().cpu().numpy() if hasattr(mol.bond_index, 'detach') else np.asarray(mol.bond_index)
                        if bi.ndim == 2 and bi.shape[1] == 2:
                            for u, v in bi.astype(int):
                                if 0 <= u < n_atoms and 0 <= v < n_atoms:
                                    if int(atom_types[v]) in allowed_neigh:
                                        neigh_lists[u].add(int(v))
                                    if int(atom_types[u]) in allowed_neigh:
                                        neigh_lists[v].add(int(u))
                    elif hasattr(mol, 'bonds') and mol.bonds is not None:
                        bi = mol.bonds.detach().cpu().numpy() if hasattr(mol.bonds, 'detach') else np.asarray(mol.bonds)
                        if bi.ndim == 2 and bi.shape[1] == 2:
                            for u, v in bi.astype(int):
                                if 0 <= u < n_atoms and 0 <= v < n_atoms:
                                    if int(atom_types[v]) in allowed_neigh:
                                        neigh_lists[u].add(int(v))
                                    if int(atom_types[u]) in allowed_neigh:
                                        neigh_lists[v].add(int(u))
                    # adjacency matrix
                    elif hasattr(mol, 'adj') and mol.adj is not None:
                        adj = mol.adj.detach().cpu().numpy() if hasattr(mol.adj, 'detach') else np.asarray(mol.adj)
                        if adj.ndim == 2 and adj.shape[0] == n_atoms:
                            for u in range(n_atoms):
                                idxs = np.flatnonzero(adj[u])
                                for v in idxs:
                                    if 0 <= v < n_atoms and v != u:
                                        if int(atom_types[v]) in allowed_neigh:
                                            neigh_lists[u].add(int(v))
                    elif hasattr(mol, 'adjacency') and mol.adjacency is not None:
                        adj = mol.adjacency.detach().cpu().numpy() if hasattr(mol.adjacency, 'detach') else np.asarray(mol.adjacency)
                        if adj.ndim == 2 and adj.shape[0] == n_atoms:
                            for u in range(n_atoms):
                                idxs = np.flatnonzero(adj[u])
                                for v in idxs:
                                    if 0 <= v < n_atoms and v != u:
                                        if int(atom_types[v]) in allowed_neigh:
                                            neigh_lists[u].add(int(v))
                # Build labels from neighbor atom types
                for u in range(n_atoms):
                    if neigh_lists[u]:
                        neigh_z = sorted(set(int(atom_types[v]) for v in neigh_lists[u]))
                        neigh_labels[u] = '-'.join(str(zv) for zv in neigh_z)
                    else:
                        neigh_labels[u] = 'none'
            except Exception:
                neigh_labels = ['none'] * n_atoms
            cols = overlaps.shape[1]
            if cols % n_atoms != 0:
                # Can't confidently split per-atom blocks
                continue
            pab = cols // n_atoms
            eff_pos = min(int(args.heatmap_pos), pab)
            # Normalize exponents globally
            norm_exps_full = normalize_exponents(exps, global_min=global_min, global_max=global_max)
            # Bin indices for each exponent row
            idx_bins = np.digitize(norm_exps_full, edges) - 1
            idx_bins = np.clip(idx_bins, 0, heat_bins - 1)
            # Log-z exponent normalization for heatmap duplicate
            if log_alpha_mean is not None and log_alpha_std is not None:
                norm_exps_logz_full = normalize_exponents_logz(exps, log_alpha_mean, log_alpha_std)
                logz_range = 4.0  # cover approximately ±4 std
                edges_logz = np.linspace(-logz_range, logz_range, heat_bins + 1)
                idx_bins_logz = np.digitize(norm_exps_logz_full, edges_logz) - 1
                idx_bins_logz = np.clip(idx_bins_logz, 0, heat_bins - 1)
            else:
                norm_exps_logz_full = np.zeros_like(norm_exps_full)
                edges_logz = None
                idx_bins_logz = None

            for a_idx in range(n_atoms):
                z = int(atom_types[a_idx])
                if include_elements and z not in include_elements:
                    continue
                start = a_idx * pab
                end = start + pab
                atom_block = overlaps[:, start:end]
                atom_block = np.abs(atom_block[:, :eff_pos])  # (E, eff_pos)

                aggr = element_aggr.get(z)
                if aggr is None:
                    aggr = _init_elem_aggr(eff_pos, heat_bins)
                    element_aggr[z] = aggr
                else:
                    eff_pos = _ensure_pos_dim(aggr, eff_pos)
                    atom_block = atom_block[:, :eff_pos]

                # Accumulate by bin
                for b in range(heat_bins):
                    mask = (idx_bins == b)
                    if not np.any(mask):
                        continue
                    aggr['sum'][b, :eff_pos] += atom_block[mask].sum(axis=0)
                    aggr['count'][b, :eff_pos] += mask.sum()
                    element_obs_counts[z] += int(mask.sum())
                # Log-z heatmap aggregation
                if edges_logz is not None and idx_bins_logz is not None:
                    aggr2 = element_aggr_logz.get(z)
                    if aggr2 is None:
                        aggr2 = _init_elem_aggr(eff_pos, heat_bins)
                        element_aggr_logz[z] = aggr2
                    else:
                        eff_pos2 = _ensure_pos_dim(aggr2, eff_pos)
                        atom_block2 = atom_block[:, :eff_pos2]
                    for b in range(heat_bins):
                        mask2 = (idx_bins_logz == b)
                        if not np.any(mask2):
                            continue
                        aggr2['sum'][b, :eff_pos] += atom_block[mask2].sum(axis=0)
                        aggr2['count'][b, :eff_pos] += mask2.sum()

                # Distribution aggregation (reservoir sampling)
                # Use raw or abs overlaps as requested
                if args.dist_abs:
                    dist_block = np.abs(overlaps[:, start:end])[:, :eff_pos]
                else:
                    dist_block = overlaps[:, start:end][:, :eff_pos]

                # Initialize or ensure pos dimension
                dist_aggr = element_dist.get(z)
                if dist_aggr is None:
                    dist_aggr = {
                        'pos_dim': eff_pos,
                        'values': [list() for _ in range(eff_pos)],
                        'counts': np.zeros(eff_pos, dtype=np.int64),
                    }
                    element_dist[z] = dist_aggr
                else:
                    # Truncate to min pos_dim if needed
                    new_dim = min(dist_aggr['pos_dim'], eff_pos)
                    if new_dim != dist_aggr['pos_dim']:
                        dist_aggr['values'] = dist_aggr['values'][:new_dim]
                        dist_aggr['counts'] = dist_aggr['counts'][:new_dim]
                        dist_aggr['pos_dim'] = new_dim
                    dist_block = dist_block[:, :dist_aggr['pos_dim']]

                # Reservoir sample per position index
                cap = int(args.dist_max_samples_per_index)
                for j in range(dist_aggr['pos_dim']):
                    col = dist_block[:, j]
                    lst = dist_aggr['values'][j]
                    # Fast path: fill up to cap
                    remaining = cap - len(lst)
                    if remaining > 0:
                        if col.size <= remaining:
                            lst.extend(col.tolist())
                            dist_aggr['counts'][j] += col.size
                            continue
                        else:
                            # Take a random subset to fill to cap
                            idxs = np.random.choice(col.size, size=remaining, replace=False)
                            lst.extend(col[idxs].tolist())
                            # Count still increases by full size (for proper reservoir odds later)
                            dist_aggr['counts'][j] += col.size
                            # Continue to reservoir replace for the remainder
                            # The elements not taken are implicitly handled by reservoir below via iteration
                    # Reservoir replacement for entire column
                    for val in col:
                        dist_aggr['counts'][j] += 1
                        c = dist_aggr['counts'][j]
                        if len(lst) < cap:
                            lst.append(float(val))
                        else:
                            r = random.randrange(c)
                            if r < cap:
                                lst[r] = float(val)

                # Per-L scatter reservoir sampling (overlap vs normalized exponent by element)
                try:
                    # Only proceed if pab has at least 25 indices (L up to 4)
                    if pab >= 25:
                        # normalized exponents per row (E,)
                        xrows = norm_exps_full
                        xrows_logz = norm_exps_logz_full if 'norm_exps_logz_full' in locals() else None

                        def _reservoir_add(series_root: dict, cat: str, X: np.ndarray, Y: np.ndarray, cap: int):
                            series_root['count'] += X.size
                            cat_series = series_root['cats'].get(cat)
                            if cat_series is None:
                                cat_series = {'x': [], 'y': [], 'count': 0}
                                series_root['cats'][cat] = cat_series
                            cat_series['count'] += X.size
                            # Fast fill
                            remaining = cap - len(cat_series['x'])
                            if remaining > 0 and X.size > 0:
                                take = min(remaining, X.size)
                                if take > 0:
                                    idxs = np.arange(X.size) if take == X.size else np.random.choice(X.size, size=take, replace=False)
                                    cat_series['x'].extend(X[idxs].tolist())
                                    cat_series['y'].extend(Y[idxs].tolist())
                                if take == X.size:
                                    return
                                # leftover for reservoir
                                keep_mask = np.ones(X.size, dtype=bool)
                                keep_mask[idxs] = False
                                X = X[~keep_mask]
                                Y = Y[~keep_mask]
                            # Reservoir for leftovers
                            for xv, yv in zip(X, Y):
                                ctot = cat_series['count']
                                r = random.randrange(ctot)
                                if len(cat_series['x']) < cap:
                                    cat_series['x'].append(float(xv))
                                    cat_series['y'].append(float(yv))
                                elif r < cap:
                                    cat_series['x'][r] = float(xv)
                                    cat_series['y'][r] = float(yv)

                        for L in range(5):
                            if L == 0:
                                startL, endL = 0, 1
                            elif L == 1:
                                startL, endL = 1, 4
                            elif L == 2:
                                startL, endL = 4, 9
                            elif L == 3:
                                startL, endL = 9, 16
                            else:
                                startL, endL = 16, 25
                            blk = overlaps[:, start:end][:, startL:endL]
                            y_all = blk.reshape(-1)
                            x_all = np.repeat(xrows, endL - startL)
                            msk = np.isfinite(x_all) & np.isfinite(y_all)
                            if not np.any(msk):
                                continue
                            x_all = x_all[msk]
                            y_all = y_all[msk]
                            cat_lbl = neigh_labels[a_idx]
                            # Original normalization
                            series = lscatter[L].get(z)
                            if series is None:
                                series = {'cats': {}, 'count': 0}
                                lscatter[L][z] = series
                            _reservoir_add(series, cat_lbl, x_all, y_all, int(args.lscatter_cap_per_series))
                            # Log-z normalization duplication
                            if xrows_logz is not None:
                                x_all_logz = np.repeat(xrows_logz, endL - startL)[msk]
                                series2 = lscatter_logz[L].get(z)
                                if series2 is None:
                                    series2 = {'cats': {}, 'count': 0}
                                    lscatter_logz[L][z] = series2
                                _reservoir_add(series2, cat_lbl, x_all_logz, y_all, int(args.lscatter_cap_per_series))
                            # Also accumulate individual points (reservoir sampled) for interactive Plotly per-L aggregated plots
                            try:
                                ag = interactive_aggr.get(L)
                                if ag is not None:
                                    cap_int = int(args.lscatter_aggregate_cap)
                                    cnt = ag['count']
                                    # iterate through individual points
                                    for xi, yv in zip(x_all.tolist(), y_all.tolist()):
                                        cnt += 1
                                        # per-point atom type (the specific atom this point came from)
                                        try:
                                            if 'atom_types' in locals() and atom_types is not None and a_idx is not None:
                                                atypes_val = str(int(atom_types[a_idx]))
                                            else:
                                                atypes_val = ''
                                        except Exception:
                                            atypes_val = ''
                                        if len(ag['x']) < cap_int:
                                            ag['x'].append(float(xi))
                                            ag['y'].append(float(yv))
                                            ag['mol_id'].append(mid_str if 'mid_str' in locals() else '')
                                            ag['atom_types'].append(atypes_val)
                                            ag['idx'].append(a_idx)
                                        else:
                                            j = random.randrange(cnt)
                                            if j < cap_int:
                                                ag['x'][j] = float(xi)
                                                ag['y'][j] = float(yv)
                                                ag['mol_id'][j] = mid_str if 'mid_str' in locals() else ''
                                                ag['atom_types'][j] = atypes_val
                                    ag['count'] = cnt
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            # Ignore per-atom processing errors
            pass

    # ========== Outlier scan and histogram ==========
    if args.outlier_scan:
        outlier_cache_path = outdir / 'outliers_summary.json'
        if args.reuse_outliers_cache and outlier_cache_path.exists():
            print(f"Reusing cached outliers summary at {outlier_cache_path}")
        else:
            print("Outlier scan: pass 1/2 — computing global mean/std over overlaps...")
            # Welford's online algorithm for mean/variance
            n_total = 0
            mean = 0.0
            M2 = 0.0
            vmin = math.inf
            vmax = -math.inf
            # First pass for stats
            for fp in progress(files, desc='Pass 1 (stats)'):
                mol = try_load_molecule(fp)
                if mol is None:
                    continue
                try:
                    overlaps = mol.overlap_int_2d.detach().cpu().numpy() if hasattr(mol.overlap_int_2d, 'detach') else np.asarray(mol.overlap_int_2d)
                    vals = np.abs(overlaps) if args.outlier_abs else overlaps
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        continue
                    # Update min/max
                    vmin = min(vmin, float(np.min(vals)))
                    vmax = max(vmax, float(np.max(vals)))
                    # Welford update in chunks to reduce precision loss
                    for x in vals.ravel():
                        n_total += 1
                        delta = float(x) - mean
                        mean += delta / n_total
                        M2 += delta * (float(x) - mean)
                except Exception:
                    continue

            if n_total < 2:
                print("Not enough data for outlier statistics.")
                std = float('nan')
            else:
                variance = M2 / (n_total - 1)
                std = math.sqrt(max(variance, 0.0))
            print(f"Global stats: count={n_total}, mean={mean:.6g}, std={std:.6g}, min={vmin:.6g}, max={vmax:.6g}")

            # Second pass: collect outliers and histogram
            print("Outlier scan: pass 2/2 — collecting outliers and histogram...")
            import heapq
            max_keep = int(args.outlier_max_save)
            # min-heap of tuples (priority, record) where priority is |z|
            heap: list[tuple[float, dict]] = []
            per_element_counts = defaultdict(int)
            per_file_counts = defaultdict(int)
            per_exp_counts = defaultdict(int)  # by exponent index or rounded normalized exponent
            # Histogram setup
            bins = int(args.hist_bins)
            if not (math.isfinite(mean) and math.isfinite(std) and std > 0):
                h_edges = np.linspace(vmin, vmax, bins + 1) if np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax else np.linspace(-1.0, 1.0, bins + 1)
            else:
                k = float(args.hist_k_sigma)
                h_edges = np.linspace(mean - k * std, mean + k * std, bins + 1)
            h_counts = np.zeros(bins, dtype=np.int64)

            for fp in progress(files, desc='Pass 2 (outliers)'):
                mol = try_load_molecule(fp)
                if mol is None:
                    continue
                try:
                    overlaps = mol.overlap_int_2d.detach().cpu().numpy() if hasattr(mol.overlap_int_2d, 'detach') else np.asarray(mol.overlap_int_2d)
                    atom_types = mol.atom_types.detach().cpu().numpy().astype(int) if hasattr(mol.atom_types, 'detach') else np.asarray(mol.atom_types).astype(int)
                    exps = mol.exponent_values.detach().cpu().numpy() if hasattr(mol.exponent_values, 'detach') else np.asarray(mol.exponent_values)
                    vals = np.abs(overlaps) if args.outlier_abs else overlaps
                    # histogram
                    c, _ = np.histogram(vals, bins=h_edges)
                    h_counts += c

                    # Outlier detection
                    if not (math.isfinite(std) and std > 0):
                        continue
                    zthr = float(args.outlier_z)
                    zscores = (vals - mean) / std
                    mask = np.abs(zscores) >= zthr
                    if not np.any(mask):
                        continue

                    # Map columns to per-atom indices
                    cols = overlaps.shape[1]
                    n_atoms = len(atom_types)
                    if n_atoms <= 0 or cols % n_atoms != 0:
                        continue
                    pab = cols // n_atoms
                    # Normalize exponents for metadata
                    try:
                        norm_exps = normalize_exponents(exps, global_min=global_min, global_max=global_max)
                    except Exception:
                        norm_exps = np.full_like(exps, np.nan, dtype=float)

                    out_idx = np.argwhere(mask)
                    for exp_idx, col_idx in out_idx:
                        a_idx = int(col_idx // pab)
                        pos_idx = int(col_idx % pab)
                        z_elem = int(atom_types[a_idx])
                        val = float(vals[exp_idx, col_idx])
                        zval = float(zscores[exp_idx, col_idx])
                        rec = {
                            'file': fp.name,
                            'mol_id': mol.id if not isinstance(mol.id, (list, tuple)) else (mol.id[0] if mol.id else None),
                            'element_Z': z_elem,
                            'atom_index': a_idx,
                            'position_index': pos_idx,
                            'exponent_index': int(exp_idx),
                            'exponent_value': float(exps[exp_idx]) if exp_idx < len(exps) else None,
                            'normalized_exponent': float(norm_exps[exp_idx]) if exp_idx < len(norm_exps) else None,
                            'overlap_value': val,
                            'zscore': zval,
                            'abs_used': bool(args.outlier_abs),
                        }
                        per_element_counts[z_elem] += 1
                        per_file_counts[fp.name] += 1
                        per_exp_counts[int(exp_idx)] += 1
                        pr = abs(zval)
                        if max_keep <= 0:
                            continue
                        if len(heap) < max_keep:
                            heapq.heappush(heap, (pr, rec))
                        else:
                            if pr > heap[0][0]:
                                heapq.heapreplace(heap, (pr, rec))

                except Exception:
                    continue

            # Prepare JSON output
            top_outliers = [rec for _, rec in sorted(heap, key=lambda t: -t[0])]
            out_summary = {
                'abs_used': bool(args.outlier_abs),
                'z_threshold': float(args.outlier_z),
                'global_mean': mean,
                'global_std': std,
                'count_total': int(n_total),
                'min_value': vmin,
                'max_value': vmax,
                'per_element_counts': dict(sorted(per_element_counts.items(), key=lambda kv: -kv[1])),
                'per_file_counts_top20': dict(list(sorted(per_file_counts.items(), key=lambda kv: -kv[1])[:20])),
                'per_exponent_index_counts': dict(sorted(per_exp_counts.items())),
                'num_outliers_saved': len(top_outliers),
                'top_outliers': top_outliers,
            }
            with open(outlier_cache_path, 'w') as f:
                json.dump(out_summary, f)
            # Also save a histogram plot
            try:
                centers = 0.5 * (h_edges[:-1] + h_edges[1:])
                plt.figure(figsize=(8,5))
                plt.bar(centers, h_counts, width=(h_edges[1]-h_edges[0])*0.9, color='steelblue', alpha=0.8)
                plt.xlabel('Overlap value' + (' (abs)' if args.outlier_abs else ''))
                plt.ylabel('Count')
                plt.title('Overlap value histogram')
                plt.tight_layout()
                plt.savefig(outdir / 'overlap_histogram.png', dpi=150)
                plt.close()
            except Exception:
                pass

            # Stacked histograms of OUTLIERS by element (atom type)
            try:
                if top_outliers:
                    # Group outlier overlap and exponent values by element Z
                    from collections import defaultdict as _dd
                    vals_by_z = _dd(list)
                    exps_by_z = _dd(list)
                    for rec in top_outliers:
                        z = rec.get('element_Z')
                        v = rec.get('overlap_value')
                        e = rec.get('exponent_value')
                        if z is None:
                            continue
                        if v is not None and np.isfinite(v):
                            vals_by_z[int(z)].append(float(v))
                        if e is not None and np.isfinite(e):
                            exps_by_z[int(z)].append(float(e))

                    # Order elements by descending count
                    elem_counts = sorted(((z, len(vs)) for z, vs in vals_by_z.items()), key=lambda kv: kv[1], reverse=True)
                    ordered_elems = [z for z, _ in elem_counts]
                    # Colors
                    cmap = plt.get_cmap('tab20')
                    colors = [cmap(i % 20) for i in range(max(1, len(ordered_elems)))]

                    def _stacked_hist(data_by_z, edges, title, xlabel, outfile, logy=False, norm_by=None):
                        if not data_by_z:
                            return
                        centers = 0.5 * (edges[:-1] + edges[1:])
                        width = (edges[1] - edges[0]) * 1.0
                        bottom = np.zeros(len(centers), dtype=np.float64)
                        plt.figure(figsize=(9, 6))
                        for i, z in enumerate(ordered_elems):
                            arr = np.asarray(data_by_z.get(z, []), dtype=float)
                            if arr.size == 0:
                                continue
                            counts, _ = np.histogram(arr, bins=edges)
                            counts = counts.astype(np.float64)
                            # Normalize by total atom occurrences for this element if provided
                            if norm_by is not None:
                                denom = float(norm_by.get(z, 0))
                                if np.isfinite(denom) and denom > 0:
                                    counts = counts / denom
                            plt.bar(centers, counts, width=width, bottom=bottom, color=colors[i], alpha=0.9, label=f'Z={z}')
                            bottom += counts
                        if logy:
                            plt.yscale('log')
                        ylab = 'Outlier count'
                        if norm_by is not None:
                            ylab = 'Normalized outlier count (per atom occurrence)'
                        if logy:
                            ylab += ' (log)'
                        plt.xlabel(xlabel)
                        plt.ylabel(ylab)
                        plt.title(title)
                        plt.legend(loc='best', fontsize=8, ncol=2)
                        plt.tight_layout()
                        plt.savefig(outfile, dpi=150)
                        plt.close()

                    # Overlap value histogram (outliers only)
                    all_vals = np.asarray([v for vs in vals_by_z.values() for v in vs], dtype=float)
                    if all_vals.size >= 2 and np.isfinite(all_vals).all():
                        vmin2 = float(np.min(all_vals))
                        vmax2 = float(np.max(all_vals))
                        if not np.isfinite(vmin2) or not np.isfinite(vmax2) or vmin2 == vmax2:
                            vmin2, vmax2 = -1.0, 1.0
                        edges_vals = np.linspace(vmin2, vmax2, 30)
                        _stacked_hist(
                            vals_by_z,
                            edges_vals,
                            title='Outlier overlaps (stacked by element)',
                            xlabel='Overlap value' + (' (abs)' if args.outlier_abs else ''),
                            outfile=outdir / 'outliers_overlap_histogram_stacked_by_element.png',
                            logy=True,
                            norm_by=element_atom_occurrences,
                        )

                    # Exponent value histogram (outliers only)
                    # use normalized exponents per element for consistent binning
                    exps_norm_by_z = {}
                    for z, es in exps_by_z.items():
                        es_arr = np.asarray(es, dtype=float)
                        if es_arr.size == 0 or not np.isfinite(es_arr).any():
                            continue
                        exps_norm_by_z[int(z)] = normalize_exponents(es_arr, global_min=global_min, global_max=global_max)
                    all_exps = np.asarray([e for es in exps_norm_by_z.values() for e in es], dtype=float)
                    if all_exps.size >= 2 and np.isfinite(all_exps).all():
                        emin = float(np.min(all_exps))
                        emax = float(np.max(all_exps))
                        if not np.isfinite(emin) or not np.isfinite(emax) or emin == emax:
                            emin, emax = 1e-3, 1.0
                        edges_exps = np.linspace(emin, emax, 19)
                        _stacked_hist(
                            exps_norm_by_z,
                            edges_exps,
                            title='Outlier exponents (stacked by element)',
                            xlabel='Normalized exponent value',
                            outfile=outdir / 'outliers_exponent_histogram_stacked_by_element.png',
                            logy=True,
                            norm_by=element_atom_occurrences,
                        )
            except Exception:
                pass
    if meta_summary['loaded'] == 0:
        print("No molecules loaded successfully. Nothing to plot.")
        return 2

    print(f"Loaded: {meta_summary['loaded']}, Skipped: {meta_summary['skipped']}")
    if meta_summary['per_atom_basis_values']:
        unique_basis = sorted(set(meta_summary['per_atom_basis_values']))
        print(f"Per-atom basis (unique): {unique_basis}")
    if meta_summary['E_values']:
        unique_E = sorted(set(meta_summary['E_values']))
        print(f"Exponents per molecule (unique): {unique_E}")

    # Concatenate across molecules (E may differ slightly; just stack) 
    norm_alpha_cat = np.concatenate(all_norm_alpha, axis=0)
    norm_alpha_logz_cat = np.concatenate(all_norm_alpha_logz, axis=0)
    mean_abs_cat = np.concatenate(all_mean_abs, axis=0)
    max_abs_cat = np.concatenate(all_max_abs, axis=0)
    std_abs_cat = np.concatenate(all_std_abs, axis=0)
    # Interactive Plotly: per-L individual overlap points vs normalized exponent (one subplot per L)
    try:
        if _plotly_available and make_subplots is not None:
            # Determine which L channels to plot (all or a specific one)
            selected = [args.interactive_l] if args.interactive_l is not None else list(range(5))
            # Check if any points collected for the selected Ls
            any_pts = any(len(interactive_aggr[L]['x']) > 0 for L in selected)
            if any_pts:
                titles = [f'L={L}' for L in selected]
                nrows = len(selected)
                fig = make_subplots(rows=nrows, cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=titles)
                for i, L in enumerate(selected):
                    ag = interactive_aggr.get(L)
                    row = i + 1
                    if not ag or len(ag['x']) == 0:
                        # add empty trace to keep subplot present
                        fig.add_trace(go.Scattergl(x=[], y=[], mode='markers', marker=dict(size=4), name=f'L={L}'), row=row, col=1)
                        fig.update_yaxes(title_text='Overlap value', row=row, col=1)
                        continue
                    xvals = np.asarray(ag['x'], dtype=float)
                    yvals = np.asarray(ag['y'], dtype=float)
                    mids = ag.get('mol_id', [])
                    atypes = ag.get('atom_types', [])
                    idxs = ag.get('idx', [])
                    # Normalize lists/arrays and guard lengths
                    n_pts = xvals.shape[0]
                    # Ensure mol_id, atom_types, idxs have length n_pts
                    try:
                        mids_arr = np.asarray(mids, dtype=object)
                    except Exception:
                        mids_arr = np.array([''] * n_pts, dtype=object)
                    if mids_arr.size != n_pts:
                        mids_arr = np.array((list(mids_arr) + [''] * n_pts)[:n_pts], dtype=object)
                    try:
                        atypes_arr = np.asarray(atypes, dtype=object)
                    except Exception:
                        atypes_arr = np.array([''] * n_pts, dtype=object)
                    if atypes_arr.size != n_pts:
                        atypes_arr = np.array((list(atypes_arr) + [''] * n_pts)[:n_pts], dtype=object)
                    try:
                        idxs_arr = np.asarray(idxs, dtype=int)
                    except Exception:
                        idxs_arr = np.full(n_pts, -1, dtype=int)
                    if idxs_arr.size != n_pts:
                        # pad or trim to match
                        tmp = list(idxs_arr)
                        tmp = (tmp + [-1] * n_pts)[:n_pts]
                        idxs_arr = np.asarray(tmp, dtype=int)
                    # Group points by atom type and add one trace per atom type so points are colored by Z
                    # Fallback: if all atom-type entries empty, create a single unlabeled trace (but include atom index)
                    non_empty_at = any(bool(a) for a in atypes_arr.tolist())
                    if not non_empty_at:
                        hover = [f"mol_id={mid}<br>atom_index={int(idx)}<br>atom_type={atypestr}<br>alpha_norm={x:.6g}<br>overlap={yv:.6g}" for mid, atypestr, idx, x, yv in zip(mids_arr.tolist(), atypes_arr.tolist(), idxs_arr.tolist(), xvals.tolist(), yvals.tolist())]
                        trace = go.Scattergl(x=xvals, y=yvals, mode='markers', marker=dict(size=4, opacity=0.7, color='#444444'), name=f'L={L}', hoverinfo='text', hovertext=hover)
                        fig.add_trace(trace, row=row, col=1)
                    else:
                        # Simple qualitative color palette (cycled)
                        palette = [
                            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
                        ]
                        unique_atypes = sorted(list(dict.fromkeys([str(a) for a in atypes_arr.tolist()])))
                        for k, at in enumerate(unique_atypes):
                            mask = (atypes_arr == at)
                            if not np.any(mask):
                                continue
                            xs = xvals[mask]
                            ys = yvals[mask]
                            mids_sub = mids_arr[mask]
                            idxs_sub = idxs_arr[mask]
                            hover_sub = [f"mol_id={mid}<br>atom_index={int(idx)}<br>atom_type={at}<br>alpha_norm={x:.6g}<br>overlap={yv:.6g}" for mid, idx, x, yv in zip(mids_sub.tolist(), idxs_sub.tolist(), xs.tolist(), ys.tolist())]
                            color = palette[k % len(palette)]
                            trace = go.Scattergl(x=xs, y=ys, mode='markers', marker=dict(size=4, opacity=0.7, color=color), name=(f'Z={at}' if at else 'Z=?'), hoverinfo='text', hovertext=hover_sub)
                            fig.add_trace(trace, row=row, col=1)
                    # set y-axis title for this row
                    fig.update_yaxes(title_text='Overlap value', row=row, col=1)
                # X axis label on bottom subplot
                fig.update_xaxes(title_text='Normalized exponent α_new', row=nrows, col=1)
                fig.update_layout(title='Overlap vs normalized exponent by L (interactive)', showlegend=True, template='plotly_white', width=1000, height=1000*nrows)
                # Output filename (include L number if single)
                if args.interactive_l is not None:
                    out_html = outdir / f'interactive_overlap_L{int(args.interactive_l)}.html'
                else:
                    out_html = outdir / 'interactive_overlap_per_L.html'
                try:
                    fig.write_html(str(out_html), include_plotlyjs='cdn')
                    print(f"Wrote interactive Plotly HTML to {out_html}")
                except Exception:
                    pass
    except Exception:
        pass
    # ================= Exponent & log-exponent histograms =================
    if all_exponents:
        try:
            exps_cat = np.concatenate(all_exponents, axis=0)
            exps_cat = exps_cat[np.isfinite(exps_cat) & (exps_cat > 0)]
            if exps_cat.size > 0:
                log_exps = np.log(exps_cat)
                bins_e = int(max(10, args.exp_hist_bins))
                # Determine ranges (robust to outliers via 0.1-99.9 percentiles)
                qlo, qhi = np.percentile(exps_cat, [0.1, 99.9]) if exps_cat.size > 100 else (exps_cat.min(), exps_cat.max())
                if not np.isfinite(qlo) or not np.isfinite(qhi) or qlo == qhi:
                    qlo, qhi = exps_cat.min(), exps_cat.max() + 1e-6
                qlo_l, qhi_l = np.percentile(log_exps, [0.1, 99.9]) if log_exps.size > 100 else (log_exps.min(), log_exps.max())
                if not np.isfinite(qlo_l) or not np.isfinite(qhi_l) or qlo_l == qhi_l:
                    qlo_l, qhi_l = log_exps.min(), log_exps.max() + 1e-6
                figH, axesH = plt.subplots(1, 3, figsize=(16, 4))
                ax1, ax2, ax3 = axesH
                ax1.hist(exps_cat, bins=bins_e, range=(qlo, qhi), color='steelblue', alpha=0.8)
                ax1.set_xlabel('Exponent value α')
                ax1.set_ylabel('Count')
                ax1.set_title('Raw exponent distribution')
                ax1.grid(alpha=0.2)
                ax2.hist(log_exps, bins=bins_e, range=(qlo_l, qhi_l), color='darkorange', alpha=0.8)
                ax2.set_xlabel('log(α)')
                ax2.set_ylabel('Count')
                ax2.set_title('Log exponent distribution')
                ax2.grid(alpha=0.2)
                # Original-normalized exponents (0–1)
                try:
                    norm_exps_01 = normalize_exponents(exps_cat, global_min=global_min, global_max=global_max)
                    ne = np.asarray(norm_exps_01, dtype=float)
                    ne = ne[np.isfinite(ne)]
                    # Keep within [0,1]
                    ne = ne[(ne >= 0.0) & (ne <= 1.0)]
                    if ne.size > 0:
                        ax3.hist(ne, bins=bins_e, range=(0.0, 1.0), color='seagreen', alpha=0.8)
                    else:
                        ax3.text(0.5, 0.5, 'No finite normalized exponents', transform=ax3.transAxes, ha='center', va='center')
                except Exception:
                    ax3.text(0.5, 0.5, 'Normalization error', transform=ax3.transAxes, ha='center', va='center')
                ax3.set_xlabel('Normalized exponent α (0–1)')
                ax3.set_ylabel('Count')
                ax3.set_title('Original normalization (0–1)')
                ax3.grid(alpha=0.2)
                figH.suptitle('Exponent, log-exponent, and normalized histograms')
                figH.tight_layout(rect=[0, 0.03, 1, 0.93])
                figH.savefig(outdir / 'exponent_value_histograms.png', dpi=150)
                plt.close(figH)
                # Store stats for summary
                exp_stats = {
                    'exp_min': float(exps_cat.min()),
                    'exp_max': float(exps_cat.max()),
                    'exp_mean': float(exps_cat.mean()),
                    'exp_std': float(exps_cat.std(ddof=0)),
                    'log_min': float(log_exps.min()),
                    'log_max': float(log_exps.max()),
                    'log_mean': float(log_exps.mean()),
                    'log_std': float(log_exps.std(ddof=0)),
                }
            else:
                exp_stats = None
        except Exception:
            exp_stats = None
    else:
        exp_stats = None

    # Plots
    plot_scatter(norm_alpha_cat, mean_abs_cat, outdir / 'scatter_norm_alpha_vs_mean_abs_overlap.png',
                 ylabel='Mean |overlap| per exponent')
    plot_scatter(norm_alpha_logz_cat, mean_abs_cat, outdir / 'scatter_logz_norm_alpha_vs_mean_abs_overlap.png', ylabel='Mean |overlap| per exponent (log-z norm)')
    plot_hexbin(norm_alpha_cat, mean_abs_cat, outdir / 'hexbin_norm_alpha_vs_mean_abs_overlap.png',
                ylabel='Mean |overlap| per exponent')
    plot_hexbin(norm_alpha_logz_cat, mean_abs_cat, outdir / 'hexbin_logz_norm_alpha_vs_mean_abs_overlap.png', ylabel='Mean |overlap| per exponent (log-z norm)')
    plot_binned(norm_alpha_cat, mean_abs_cat, outdir / 'binned_norm_alpha_vs_mean_abs_overlap.png',
                ylabel='Mean |overlap| per exponent', bins=18)
    plot_binned(norm_alpha_logz_cat, mean_abs_cat, outdir / 'binned_logz_norm_alpha_vs_mean_abs_overlap.png', ylabel='Mean |overlap| per exponent (log-z norm)', bins=18)

    # Additional: visualize spread via max and std
    plot_scatter(norm_alpha_cat, max_abs_cat, outdir / 'scatter_norm_alpha_vs_max_abs_overlap.png',
                 ylabel='Max |overlap| per exponent')
    plot_scatter(norm_alpha_logz_cat, max_abs_cat, outdir / 'scatter_logz_norm_alpha_vs_max_abs_overlap.png', ylabel='Max |overlap| per exponent (log-z norm)')
    plot_binned(norm_alpha_cat, std_abs_cat, outdir / 'binned_norm_alpha_vs_std_abs_overlap.png',
                ylabel='Std |overlap| per exponent', bins=18)
    plot_binned(norm_alpha_logz_cat, std_abs_cat, outdir / 'binned_logz_norm_alpha_vs_std_abs_overlap.png', ylabel='Std |overlap| per exponent (log-z norm)', bins=18)

    # Per-L histograms (overlap values grouped by L=0..4) with per-L x-range
    try:
        bins_L = int(args.hist_bins)
        cum = [1, 4, 9, 16, 25, 36, 49]
        # First pass: per-L min/max
        L_min = [math.inf] * 5
        L_max = [-math.inf] * 5
        for fp in progress(files, desc='Per-L hist (range pass)'):
            mol = try_load_molecule(fp)
            if mol is None:
                continue
            try:
                overlaps = mol.overlap_int_2d.detach().cpu().numpy() if hasattr(mol.overlap_int_2d, 'detach') else np.asarray(mol.overlap_int_2d)
                atom_types = mol.atom_types.detach().cpu().numpy().astype(int) if hasattr(mol.atom_types, 'detach') else np.asarray(mol.atom_types).astype(int)
                n_atoms = len(atom_types)
                if n_atoms <= 0:
                    continue
                cols = overlaps.shape[1]
                if cols % n_atoms != 0:
                    continue
                pab = cols // n_atoms
                if pab < 25:
                    continue
                for a_idx in range(n_atoms):
                    base = a_idx * pab
                    for L in range(5):
                        startL = 0 if L == 0 else cum[L-1]
                        endL = cum[L]
                        if endL > pab:
                            continue
                        block = overlaps[:, base + startL: base + endL]
                        vals = block.ravel()
                        vals = vals[np.isfinite(vals)]
                        if vals.size == 0:
                            continue
                        vminL = float(np.min(vals))
                        vmaxL = float(np.max(vals))
                        if vminL < L_min[L]:
                            L_min[L] = vminL
                        if vmaxL > L_max[L]:
                            L_max[L] = vmaxL
            except Exception:
                continue
        # Prepare edges per L
        edges_per_L: dict[int, np.ndarray] = {}
        for L in range(5):
            vminL, vmaxL = L_min[L], L_max[L]
            if not (np.isfinite(vminL) and np.isfinite(vmaxL)) or vminL == math.inf or vmaxL == -math.inf:
                # fallback default range
                vminL, vmaxL = -1.0, 1.0
            if vminL == vmaxL:
                # expand symmetric window
                delta = 0.5 if vminL == 0 else 0.1 * abs(vminL)
                vminL -= delta
                vmaxL += delta
            edges_per_L[L] = np.linspace(vminL, vmaxL, bins_L + 1)
        # Second pass: counts per L with own edges
        counts_by_L = {L: np.zeros(bins_L, dtype=np.int64) for L in range(5)}
        for fp in progress(files, desc='Per-L hist (count pass)'):
            mol = try_load_molecule(fp)
            if mol is None:
                continue
            try:
                overlaps = mol.overlap_int_2d.detach().cpu().numpy() if hasattr(mol.overlap_int_2d, 'detach') else np.asarray(mol.overlap_int_2d)
                atom_types = mol.atom_types.detach().cpu().numpy().astype(int) if hasattr(mol.atom_types, 'detach') else np.asarray(mol.atom_types).astype(int)
                n_atoms = len(atom_types)
                if n_atoms <= 0:
                    continue
                cols = overlaps.shape[1]
                if cols % n_atoms != 0:
                    continue
                pab = cols // n_atoms
                if pab < 25:
                    continue
                for a_idx in range(n_atoms):
                    base = a_idx * pab
                    for L in range(5):
                        edgesL = edges_per_L[L]
                        startL = 0 if L == 0 else cum[L-1]
                        endL = cum[L]
                        if endL > pab:
                            continue
                        block = overlaps[:, base + startL: base + endL]
                        vals = block.ravel()
                        vals = vals[np.isfinite(vals)]
                        if vals.size == 0:
                            continue
                        c, _ = np.histogram(vals, bins=edgesL)
                        counts_by_L[L] += c
            except Exception:
                continue
        # Plot subplots with individual x ranges
        fig, axes = plt.subplots(1, 5, figsize=(20, 3.8), sharey=True)
        L_labels = ['L=0 (s)', 'L=1 (p)', 'L=2 (d)', 'L=3 (f)', 'L=4 (g)']
        for L in range(5):
            ax = axes[L]
            edgesL = edges_per_L[L]
            centers = 0.5 * (edgesL[:-1] + edgesL[1:])
            width = (edgesL[1] - edgesL[0]) * 1.0
            ax.bar(centers, counts_by_L[L], width=width, color='steelblue', alpha=0.85)
            ax.set_title(L_labels[L])
            ax.set_yscale('log')
            ax.set_xlabel('Overlap value')
            if L == 0:
                ax.set_ylabel('Count (log)')
        fig.suptitle('Overlap value histograms grouped by L (individual x-range)')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(outdir / 'overlap_histograms_by_L.png', dpi=150)
        plt.close(fig)
    except Exception:
        pass

    # Per-L scatter plots: overlap vs normalized exponent, colored by atom type
    try:
        # Build a list of elements ordered by total sampled points across all L
        counts_total = defaultdict(int)
        for L in range(5):
            for z, series in lscatter[L].items():
                counts_total[z] += int(series.get('count', 0))
        if counts_total:
            ordered = sorted(counts_total.items(), key=lambda kv: kv[1], reverse=True)
            top_cols = [z for z, _ in ordered[:5]]
            # Determine top neighbor categories across selected elements
            cat_counts = defaultdict(int)
            for L in range(5):
                for z in top_cols:
                    series = lscatter[L].get(z)
                    if not series:
                        continue
                    for cat_lbl, cat_series in series['cats'].items():
                        cat_counts[cat_lbl] += len(cat_series.get('x', []))
            top_cats = [c for c, _ in sorted(cat_counts.items(), key=lambda kv: kv[1], reverse=True)[: int(args.lscatter_max_neigh_cats)]]
            # Color map for neighbor categories
            cmap = plt.get_cmap('tab20')
            cat_color = {cat: cmap(i % 20) for i, cat in enumerate(top_cats)}
            other_color = (0.7, 0.7, 0.7, 0.6)
            # Create a 5x5 grid: rows=L (0..4), cols=top 5 elements
            nrows, ncols = 5, 5
            # Robust per-L y limits across all selected elements (1st-99th percentile with padding)
            row_y_limits = {}
            for L in range(5):
                all_y = []
                for z in top_cols:
                    series = lscatter[L].get(z)
                    if not series or not series.get('cats'):
                        continue
                    for cat_series in series['cats'].values():
                        ys = cat_series.get('y')
                        if ys:
                            all_y.extend([yv for yv in ys if np.isfinite(yv)])
                if all_y:
                    arr = np.asarray(all_y, dtype=float)
                    q1, q99 = np.percentile(arr, [1.0, 99.0])
                    if not np.isfinite(q1) or not np.isfinite(q99):
                        continue
                    if q1 == q99:
                        q1 -= 0.5
                        q99 += 0.5
                    span = q99 - q1
                    pad = 0.05 * span
                    row_y_limits[L] = (q1 - pad, q99 + pad)
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows), sharex=True, sharey=False)
            L_labels = ['L=0 (s)', 'L=1 (p)', 'L=2 (d)', 'L=3 (f)', 'L=4 (g)']
            # Precompute bin edges for stats lines if needed
            if args.lscatter_stats_lines:
                stats_bins = max(5, int(args.lscatter_stats_bins))
                stats_edges = np.linspace(0.0, 1.0, stats_bins + 1)
                stats_centers = 0.5 * (stats_edges[:-1] + stats_edges[1:])
            else:
                stats_edges = stats_centers = None  # type: ignore
            # Helper for stats overlay to reduce duplicated try/except blocks
            def _maybe_stats_overlay(ax, series, iL, j):
                if not (args.lscatter_stats_lines and stats_edges is not None):
                    return
                try:
                    if not series or not series.get('cats'):
                        return
                    # Concatenate all categories for this (L,Z)
                    x_all = np.concatenate([cat_series['x'] for cat_series in series['cats'].values() if cat_series['x']], dtype=float)
                    y_all = np.concatenate([cat_series['y'] for cat_series in series['cats'].values() if cat_series['y']], dtype=float)
                    if x_all.size == 0 or y_all.size != x_all.size:
                        return
                    idx = np.digitize(x_all, stats_edges) - 1
                    valid = (idx >= 0) & (idx < stats_edges.size - 1)
                    if not np.any(valid):
                        return
                    idx = idx[valid]
                    yv = y_all[valid]
                    # Aggregate using bincount
                    nb = stats_edges.size - 1
                    sum_y = np.bincount(idx, weights=yv, minlength=nb)
                    sum_y2 = np.bincount(idx, weights=yv*yv, minlength=nb)
                    counts = np.bincount(idx, minlength=nb)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        mean_line = sum_y / counts
                        var_line = (sum_y2 / counts) - mean_line**2
                    mean_line[counts <= 0] = np.nan
                    var_line[counts <= 1] = np.nan
                    std_line = np.sqrt(np.clip(var_line, 0.0, None))
                    ax.plot(stats_centers, mean_line, color='black', linewidth=1.0, label='mean' if (iL==0 and j==0) else None)
                    ax.fill_between(stats_centers, mean_line - std_line, mean_line + std_line, color='black', alpha=0.12, linewidth=0)
                except Exception as e:  # pragma: no cover - defensive
                    print(f"[WARN] stats overlay failed for subplot (L={L}, j={j}): {e}")

            for iL, L in enumerate(range(5)):
                for j, z in enumerate(top_cols):
                    ax = axes[iL, j]
                    series = lscatter[L].get(z)
                    if series and series.get('cats'):
                        # Compute per-category alpha map if requested
                        alpha_map = None
                        base_alpha = 0.35
                        if args.lscatter_weight_by_freq:
                            freqs = {}
                            for cat_lbl, cat_series in series['cats'].items():
                                cnt = int(cat_series.get('count', 0))
                                if cnt <= 0:
                                    cnt = len(cat_series.get('x', []))
                                freqs[cat_lbl] = max(1, cnt)
                            w_raw = {k: 1.0 / math.sqrt(v) for k, v in freqs.items()} if freqs else {}
                            if w_raw:
                                wmax = max(w_raw.values())
                                if wmax <= 0:
                                    wmax = 1.0
                                # Clamp to avoid near-zero visibility
                                min_frac = 0.15
                                alpha_map = {k: max(base_alpha * min_frac, base_alpha * (w / wmax)) for k, w in w_raw.items()}
                        X = []
                        Y = []
                        Cc = []
                        Lbl = []
                        for cat_lbl, cat_series in series['cats'].items():
                            if not cat_series['x']:
                                continue
                            col = cat_color.get(cat_lbl, other_color)
                            npts = len(cat_series['x'])
                            X.extend(cat_series['x'])
                            Y.extend(cat_series['y'])
                            Cc.extend([col] * npts)
                            Lbl.extend([cat_lbl] * npts)
                        if X:
                            perm = np.random.permutation(len(X))
                            X = np.asarray(X, dtype=float)[perm]
                            Y = np.asarray(Y, dtype=float)[perm]
                            Cc_perm = [Cc[k] for k in perm]
                            if args.lscatter_weight_by_freq and alpha_map is not None:
                                Lbl_perm = [Lbl[k] for k in perm]
                                Cc_adj = []
                                for col, lbl in zip(Cc_perm, Lbl_perm):
                                    r, g, b = col[0], col[1], col[2]
                                    a = alpha_map.get(lbl, base_alpha)
                                    Cc_adj.append((r, g, b, a))
                                ax.scatter(X, Y, s=0.5, edgecolors='none', c=Cc_adj)
                            else:
                                ax.scatter(X, Y, s=0.5, alpha=base_alpha, edgecolors='none', c=Cc_perm)
                            if L in row_y_limits:
                                ax.set_ylim(row_y_limits[L])
                            # Stats overlay (mean and ±std) using concatenated sampled points across categories
                            _maybe_stats_overlay(ax, series, iL, j)
                    else:
                        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=8, color='gray')
                        if L in row_y_limits:
                            ax.set_ylim(row_y_limits[L])
                    if iL == 0:
                        ax.set_title(f'Z={z}')
                    if j == 0:
                        ax.set_ylabel(f"{L_labels[iL]}\nOverlap value")
                    if iL == nrows - 1:
                        ax.set_xlabel('Normalized exponent α_new')
            # Global legend for neighbor categories
            from matplotlib.lines import Line2D
            handles = [Line2D([0],[0], marker='o', color='none', markerfacecolor=cat_color[c], markeredgecolor='none', markersize=6, label=c) for c in top_cats]
            if handles:
                fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=8, title='Neighbor types')
            fig.suptitle('Overlap vs normalized exponent by L and element (colored by neighbor types)')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(outdir / 'overlap_vs_norm_exponent_LxElement_grid.png', dpi=150)
            plt.close(fig)

            # Second version with high alpha to emphasize density
            fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows), sharex=True, sharey=False)
            for iL, L in enumerate(range(5)):
                for j, z in enumerate(top_cols):
                    ax = axes2[iL, j]
                    series = lscatter[L].get(z)
                    if series and series.get('cats'):
                        # Compute per-category alpha map if requested (base alpha 1.0)
                        alpha_map = None
                        base_alpha2 = 1.0
                        if args.lscatter_weight_by_freq:
                            freqs = {}
                            for cat_lbl, cat_series in series['cats'].items():
                                cnt = int(cat_series.get('count', 0))
                                if cnt <= 0:
                                    cnt = len(cat_series.get('x', []))
                                freqs[cat_lbl] = max(1, cnt)
                            w_raw = {k: 1.0 / math.sqrt(v) for k, v in freqs.items()} if freqs else {}
                            if w_raw:
                                wmax = max(w_raw.values())
                                if wmax <= 0:
                                    wmax = 1.0
                                min_frac = 0.1
                                alpha_map = {k: max(base_alpha2 * min_frac, base_alpha2 * (w / wmax)) for k, w in w_raw.items()}
                        X = []
                        Y = []
                        Cc = []
                        Lbl = []
                        for cat_lbl, cat_series in series['cats'].items():
                            if not cat_series['x']:
                                continue
                            col = cat_color.get(cat_lbl, other_color)
                            npts = len(cat_series['x'])
                            X.extend(cat_series['x'])
                            Y.extend(cat_series['y'])
                            Cc.extend([col] * npts)
                            Lbl.extend([cat_lbl] * npts)
                        if X:
                            perm = np.random.permutation(len(X))
                            X = np.asarray(X, dtype=float)[perm]
                            Y = np.asarray(Y, dtype=float)[perm]
                            Cc_perm = [Cc[k] for k in perm]
                            if args.lscatter_weight_by_freq and alpha_map is not None:
                                Lbl_perm = [Lbl[k] for k in perm]
                                Cc_adj = []
                                for col, lbl in zip(Cc_perm, Lbl_perm):
                                    r, g, b = col[0], col[1], col[2]
                                    a = alpha_map.get(lbl, base_alpha2)
                                    Cc_adj.append((r, g, b, a))
                                ax.scatter(X, Y, s=0.5, edgecolors='none', c=Cc_adj)
                            else:
                                ax.scatter(X, Y, s=0.5, alpha=base_alpha2, edgecolors='none', c=Cc_perm)
                            if L in row_y_limits:
                                ax.set_ylim(row_y_limits[L])
                            # Stats overlay on high-alpha version too (optional)
                            _maybe_stats_overlay(ax, series, iL, j)
                    else:
                        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=8, color='gray')
                        if L in row_y_limits:
                            ax.set_ylim(row_y_limits[L])
                    if iL == 0:
                        ax.set_title(f'Z={z}')
                    if j == 0:
                        ax.set_ylabel(f"{L_labels[iL]}\nOverlap value")
                    if iL == nrows - 1:
                        ax.set_xlabel('Normalized exponent α_new')
            from matplotlib.lines import Line2D as _L2D
            handles2 = [_L2D([0],[0], marker='o', color='none', markerfacecolor=cat_color[c], markeredgecolor='none', markersize=6, label=c) for c in top_cats]
            if handles2:
                fig2.legend(handles=handles2, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=8, title='Neighbor types')
            fig2.suptitle('Overlap vs normalized exponent by L and element (high-alpha to show density)')
            fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig2.savefig(outdir / 'overlap_vs_norm_exponent_LxElement_grid_highalpha.png', dpi=150)
            plt.close(fig2)
            # ============ Duplicate per-L element grids with log-z normalization ============
            try:
                # Reuse earlier ordering and categories
                if lscatter_logz:
                    figLZ, axesLZ = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows), sharex=True, sharey=False)
                    # Determine global log-z x-range from sampled points
                    logz_all = []
                    for L in range(5):
                        for z in top_cols:
                            series = lscatter_logz[L].get(z)
                            if not series or not series.get('cats'):
                                continue
                            for cs in series['cats'].values():
                                if cs['x']:
                                    logz_all.extend([v for v in cs['x'] if np.isfinite(v)])
                    if logz_all:
                        arr_lz = np.asarray(logz_all, dtype=float)
                        q1_lz, q99_lz = np.percentile(arr_lz, [1, 99])
                        span_lz = q99_lz - q1_lz if q99_lz > q1_lz else 1.0
                        pad_lz = 0.05 * span_lz
                        xlim_logz = (q1_lz - pad_lz, q99_lz + pad_lz)
                    else:
                        xlim_logz = (-4, 4)
                    # Y-axis hard clip configuration for log-z overlap plots
                    Y_CLIP = 10.0
                    clip_counts = {L: {z: 0 for z in top_cols} for L in range(5)}
                    total_clip = 0
                    for iL, L in enumerate(range(5)):
                        for j, z in enumerate(top_cols):
                            ax = axesLZ[iL, j]
                            series = lscatter_logz[L].get(z)
                            if series and series.get('cats'):
                                X = []
                                Y = []
                                for cat_series in series['cats'].values():
                                    if cat_series['x']:
                                        X.extend(cat_series['x'])
                                        Y.extend(cat_series['y'])
                                if X:
                                    perm = np.random.permutation(len(X))
                                    X = np.asarray(X, dtype=float)[perm]
                                    Y = np.asarray(Y, dtype=float)[perm]
                                    # Count points that would be outside clip
                                    outside = (Y > Y_CLIP) | (Y < -Y_CLIP)
                                    n_out = int(np.sum(outside))
                                    if n_out > 0:
                                        clip_counts[L][z] = n_out
                                        total_clip += n_out
                                        # Clip values for plotting
                                        Y = np.clip(Y, -Y_CLIP, Y_CLIP)
                                    ax.scatter(X, Y, s=0.5, alpha=0.35, edgecolors='none', c='steelblue')
                                if L in row_y_limits:
                                    # Apply dynamic limits but respect hard clip cap
                                    ylo, yhi = row_y_limits[L]
                                    ylo = max(ylo, -Y_CLIP)
                                    yhi = min(yhi, Y_CLIP)
                                    ax.set_ylim((ylo, yhi))
                                else:
                                    ax.set_ylim((-Y_CLIP, Y_CLIP))
                            else:
                                ax.text(0.5,0.5,'No data', transform=ax.transAxes, ha='center', va='center', fontsize=8, color='gray')
                                ax.set_ylim((-Y_CLIP, Y_CLIP))
                            ax.set_xlim(xlim_logz)
                            if iL == 0:
                                ax.set_title(f'Z={z}')
                            if j == 0:
                                ax.set_ylabel(f"{L_labels[iL]}\nOverlap value (clipped ±{Y_CLIP})")
                            if iL == nrows - 1:
                                ax.set_xlabel('Log-z normalized exponent')
                    title_extra = '' if total_clip == 0 else f'  (clipped points hidden: {total_clip})'
                    figLZ.suptitle('Overlap vs log-z exponent by L and element' + title_extra)
                    figLZ.tight_layout(rect=[0,0.03,1,0.95])
                    figLZ.savefig(outdir / 'overlap_vs_logz_exponent_LxElement_grid.png', dpi=150)
                    plt.close(figLZ)
            except Exception as e:
                print(f"[WARN] Failed generating log-z per-L element grids: {e}")

            # ================= Local bin-wise z-score normalization plots =================
            if args.lscatter_local_z:
                try:
                    z_bins = max(20, int(args.lscatter_local_z_bins))
                    z_edges = np.linspace(0.0, 1.0, z_bins + 1)
                    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
                    # Precompute per (L,Z) bin means/stds using concatenated categories
                    stats_local = {L: {} for L in range(5)}  # L -> Z -> dict(mean, std, counts)
                    for L in range(5):
                        for z in top_cols:
                            series = lscatter[L].get(z)
                            if not series or not series.get('cats'):
                                continue
                            try:
                                x_all = np.concatenate([cs['x'] for cs in series['cats'].values() if cs['x']], dtype=float)
                                y_all = np.concatenate([cs['y'] for cs in series['cats'].values() if cs['y']], dtype=float)
                            except ValueError:
                                continue
                            if x_all.size == 0 or y_all.size != x_all.size:
                                continue
                            idx = np.digitize(x_all, z_edges) - 1
                            valid = (idx >= 0) & (idx < z_bins)
                            if not np.any(valid):
                                continue
                            idx = idx[valid]
                            yv = y_all[valid]
                            sum_y = np.bincount(idx, weights=yv, minlength=z_bins)
                            sum_y2 = np.bincount(idx, weights=yv*yv, minlength=z_bins)
                            counts = np.bincount(idx, minlength=z_bins)
                            with np.errstate(divide='ignore', invalid='ignore'):
                                mean_line = sum_y / counts
                                var_line = (sum_y2 / counts) - mean_line**2
                            mean_line[counts == 0] = np.nan
                            var_line[counts <= 1] = np.nan
                            std_line = np.sqrt(np.clip(var_line, 0.0, None))
                            stats_local[L][z] = {
                                'mean': mean_line,
                                'std': std_line,
                                'counts': counts,
                            }
                    # Build z-scatter data: per point assign its bin z-score if stats valid
                    zscatter = {L: {z: {'x': [], 'z': [], 'count': 0} for z in top_cols} for L in range(5)}
                    min_count_threshold = 5
                    for L in range(5):
                        for z in top_cols:
                            series = lscatter[L].get(z)
                            if not series or not series.get('cats') or z not in stats_local[L]:
                                continue
                            loc_stats = stats_local[L][z]
                            mean_arr = loc_stats['mean']
                            std_arr = loc_stats['std']
                            counts_arr = loc_stats['counts']
                            if mean_arr is None or std_arr is None:
                                continue
                            for cat_lbl, cat_series in series['cats'].items():
                                xs = cat_series.get('x')
                                ys = cat_series.get('y')
                                if not xs or not ys:
                                    continue
                                xs_arr = np.asarray(xs, dtype=float)
                                ys_arr = np.asarray(ys, dtype=float)
                                if xs_arr.size != ys_arr.size or xs_arr.size == 0:
                                    continue
                                b_idx = np.digitize(xs_arr, z_edges) - 1
                                valid = (b_idx >= 0) & (b_idx < z_bins)
                                if not np.any(valid):
                                    continue
                                xs_arr = xs_arr[valid]
                                ys_arr = ys_arr[valid]
                                b_idx = b_idx[valid]
                                # Gather stats and compute z
                                m = mean_arr[b_idx]
                                s = std_arr[b_idx]
                                cts = counts_arr[b_idx]
                                good = np.isfinite(m) & np.isfinite(s) & (s > 0) & (cts >= min_count_threshold)
                                if not np.any(good):
                                    continue
                                xs_g = xs_arr[good]
                                ys_g = ys_arr[good]
                                m_g = m[good]
                                s_g = s[good]
                                zvals = (ys_g - m_g) / s_g
                                zscatter[L][z]['x'].extend(xs_g.tolist())
                                zscatter[L][z]['z'].extend(zvals.tolist())
                                zscatter[L][z]['count'] += len(zvals)
                    # Compute robust per-L limits for z
                    z_row_limits = {}
                    for L in range(5):
                        allz = []
                        for z in top_cols:
                            zs = zscatter[L].get(z, {}).get('z')
                            if zs:
                                allz.extend([v for v in zs if np.isfinite(v)])
                        if allz:
                            arr = np.asarray(allz, dtype=float)
                            q1, q99 = np.percentile(arr, [1.0, 99.0])
                            if not np.isfinite(q1) or not np.isfinite(q99):
                                continue
                            if q1 == q99:
                                q1 -= 0.5
                                q99 += 0.5
                            span = q99 - q1
                            pad = 0.05 * span
                            z_row_limits[L] = (q1 - pad, q99 + pad)
                    # Plot local z normalized scatter (base alpha)
                    figz1, axz1 = plt.subplots(5, 5, figsize=(4.0 * 5, 3.0 * 5), sharex=True, sharey=False)
                    for iL, L in enumerate(range(5)):
                        for j, z in enumerate(top_cols):
                            ax = axz1[iL, j]
                            series_z = zscatter[L].get(z)
                            if series_z and series_z['x']:
                                X = np.asarray(series_z['x'], dtype=float)
                                ZV = np.asarray(series_z['z'], dtype=float)
                                perm = np.random.permutation(len(X))
                                X = X[perm]
                                ZV = ZV[perm]
                                ax.scatter(X, ZV, s=0.5, alpha=0.35, edgecolors='none', c='steelblue')
                                if L in z_row_limits:
                                    ax.set_ylim(z_row_limits[L])
                            else:
                                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=8, color='gray')
                            ax.axhline(0.0, color='black', linewidth=0.6, alpha=0.6)
                            if iL == 0:
                                ax.set_title(f'Z={z}')
                            if j == 0:
                                ax.set_ylabel(f'L={L}\nLocal z')
                            if iL == 4:
                                ax.set_xlabel('Normalized exponent α_new')
                    figz1.suptitle('Local bin-wise z = (O - μ_bin)/σ_bin vs normalized exponent (by L and element)')
                    figz1.tight_layout(rect=[0, 0.03, 1, 0.95])
                    figz1.savefig(outdir / 'overlap_localZ_vs_norm_exponent_LxElement_grid.png', dpi=150)
                    plt.close(figz1)
                    # High alpha version
                    figz2, axz2 = plt.subplots(5, 5, figsize=(4.0 * 5, 3.0 * 5), sharex=True, sharey=False)
                    for iL, L in enumerate(range(5)):
                        for j, z in enumerate(top_cols):
                            ax = axz2[iL, j]
                            series_z = zscatter[L].get(z)
                            if series_z and series_z['x']:
                                X = np.asarray(series_z['x'], dtype=float)
                                ZV = np.asarray(series_z['z'], dtype=float)
                                perm = np.random.permutation(len(X))
                                X = X[perm]
                                ZV = ZV[perm]
                                ax.scatter(X, ZV, s=0.5, alpha=1.0, edgecolors='none', c='steelblue')
                                if L in z_row_limits:
                                    ax.set_ylim(z_row_limits[L])
                            else:
                                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=8, color='gray')
                            ax.axhline(0.0, color='black', linewidth=0.6, alpha=0.6)
                            if iL == 0:
                                ax.set_title(f'Z={z}')
                            if j == 0:
                                ax.set_ylabel(f'L={L}\nLocal z')
                            if iL == 4:
                                ax.set_xlabel('Normalized exponent α_new')
                    figz2.suptitle('Local bin-wise z (high-alpha) = (O - μ_bin)/σ_bin vs normalized exponent (by L and element)')
                    figz2.tight_layout(rect=[0, 0.03, 1, 0.95])
                    figz2.savefig(outdir / 'overlap_localZ_vs_norm_exponent_LxElement_grid_highalpha.png', dpi=150)
                    plt.close(figz2)
                    # Duplicate local bin-wise z using log-z normalized exponent axis (same binning range ±4)
                    try:
                        z_edges_logz = np.linspace(-4.0, 4.0, z_bins + 1)
                        zscatter_logz = {L: {z: {'x': [], 'z': [], 'count': 0} for z in top_cols} for L in range(5)}
                        for L in range(5):
                            for z in top_cols:
                                series = lscatter_logz[L].get(z)
                                if not series or not series.get('cats'):
                                    continue
                                x_all = np.concatenate([cs['x'] for cs in series['cats'].values() if cs['x']], dtype=float)
                                y_all = np.concatenate([cs['y'] for cs in series['cats'].values() if cs['y']], dtype=float)
                                if x_all.size == 0:
                                    continue
                                idx = np.digitize(x_all, z_edges_logz) - 1
                                valid = (idx >= 0) & (idx < z_bins)
                                if not np.any(valid):
                                    continue
                                idxv = idx[valid]
                                yv = y_all[valid]
                                sum_y = np.bincount(idxv, weights=yv, minlength=z_bins)
                                sum_y2 = np.bincount(idxv, weights=yv*yv, minlength=z_bins)
                                counts = np.bincount(idxv, minlength=z_bins)
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    mean_line = sum_y / counts
                                    var_line = (sum_y2 / counts) - mean_line**2
                                mean_line[counts == 0] = np.nan
                                var_line[counts <= 1] = np.nan
                                std_line = np.sqrt(np.clip(var_line, 0.0, None))
                                # Assign per point z
                                idx_points = idxv
                                m = mean_line[idx_points]
                                s = std_line[idx_points]
                                good = np.isfinite(m) & np.isfinite(s) & (s > 0)
                                if not np.any(good):
                                    continue
                                xv_good = x_all[valid][good]
                                yv_good = yv[good]
                                zvals = (yv_good - m[good]) / s[good]
                                zscatter_logz[L][z]['x'].extend(xv_good.tolist())
                                zscatter_logz[L][z]['z'].extend(zvals.tolist())
                                zscatter_logz[L][z]['count'] += len(zvals)
                        # Plot
                        figLZ1, axLZ1 = plt.subplots(5, 5, figsize=(4.0*5, 3.0*5), sharex=True, sharey=False)
                        # derive x-range for log-z local z plots
                        logz_all2 = []
                        for L in range(5):
                            for z in top_cols:
                                ser = lscatter_logz[L].get(z)
                                if ser and ser.get('cats'):
                                    for cs in ser['cats'].values():
                                        if cs['x']:
                                            logz_all2.extend([v for v in cs['x'] if np.isfinite(v)])
                        if logz_all2:
                            arr2 = np.asarray(logz_all2, dtype=float)
                            q1b, q99b = np.percentile(arr2, [1, 99])
                            spanb = q99b - q1b if q99b > q1b else 1.0
                            padb = 0.05 * spanb
                            xlim_lz_local = (q1b - padb, q99b + padb)
                        else:
                            xlim_lz_local = (-4, 4)
                        Y_CLIP = 10.0
                        total_clip_lz = 0
                        for iL, L in enumerate(range(5)):
                            for j, z in enumerate(top_cols):
                                ax = axLZ1[iL, j]
                                series_z = zscatter_logz[L].get(z)
                                clipped_here = 0
                                if series_z and series_z['x']:
                                    X = np.asarray(series_z['x'], dtype=float)
                                    ZV = np.asarray(series_z['z'], dtype=float)
                                    perm = np.random.permutation(len(X))
                                    X = X[perm]; ZV = ZV[perm]
                                    outside = (ZV > Y_CLIP) | (ZV < -Y_CLIP)
                                    clipped_here = int(np.sum(outside))
                                    if clipped_here > 0:
                                        total_clip_lz += clipped_here
                                        ZV = np.clip(ZV, -Y_CLIP, Y_CLIP)
                                    ax.scatter(X, ZV, s=0.5, alpha=0.35, edgecolors='none', c='steelblue')
                                else:
                                    ax.text(0.5,0.5,'No data', transform=ax.transAxes, ha='center', va='center', fontsize=8, color='gray')
                                ax.axhline(0.0, color='black', linewidth=0.6, alpha=0.6)
                                ax.set_xlim(xlim_lz_local)
                                ax.set_ylim((-Y_CLIP, Y_CLIP))
                                title_lbl = f'Z={z}' if iL == 0 else ax.get_title()
                                if clipped_here > 0 and iL == 0:
                                    title_lbl += f'\nclipped {clipped_here}'
                                if iL == 0:
                                    ax.set_title(title_lbl, fontsize=9)
                                if j == 0:
                                    ax.set_ylabel(f'L={L}\nLocal z (clipped ±{Y_CLIP})')
                                if iL == 4:
                                    ax.set_xlabel('Log-z normalized exponent')
                        t_extra_lz = '' if total_clip_lz == 0 else f'  (total clipped points hidden: {total_clip_lz})'
                        figLZ1.suptitle('Local bin-wise z vs log-z exponent (by L and element)' + t_extra_lz)
                        figLZ1.tight_layout(rect=[0,0.03,1,0.95])
                        figLZ1.savefig(outdir / 'overlap_localZ_vs_logz_exponent_LxElement_grid.png', dpi=150)
                        plt.close(figLZ1)
                    except Exception as e2:
                        print(f"[WARN] Failed generating log-z local z grids: {e2}")
                except Exception as e:
                    print(f"[WARN] Failed generating local z-score lscatter plots: {e}")

            # Skip the detailed per-element neighbor plots when stats lines are requested to reduce clutter & avoid extra failures.
            if not args.lscatter_stats_lines:
                # Per-element figures: rows=L, cols=neighbor categories (no color coding)
                try:
                    elems = sorted({z for L in range(5) for z in lscatter[L].keys()})
                    for z in elems:
                        cat_counts_z = defaultdict(int)
                        for L in range(5):
                            series = lscatter[L].get(z)
                            if not series or not series.get('cats'):
                                continue
                            for cat_lbl, cat_series in series['cats'].items():
                                cat_counts_z[cat_lbl] += len(cat_series.get('x', []))
                        if not cat_counts_z:
                            continue
                        top_cats_z = [c for c, _ in sorted(cat_counts_z.items(), key=lambda kv: kv[1], reverse=True)[: int(args.lscatter_max_neigh_cats)]]
                        ncols_e = max(1, len(top_cats_z))
                        nrows_e = 5
                        figz, axesz = plt.subplots(nrows_e, ncols_e, figsize=(3.5 * ncols_e, 3.0 * nrows_e), sharex=True, sharey=False)
                        row_y_limits_z = {}
                        for L in range(5):
                            seriesL = lscatter[L].get(z)
                            if not seriesL or not seriesL.get('cats'):
                                continue
                            all_y = []
                            for cat_lbl in top_cats_z:
                                cat_series = seriesL['cats'].get(cat_lbl)
                                if cat_series and cat_series.get('y'):
                                    all_y.extend([yv for yv in cat_series['y'] if np.isfinite(yv)])
                            if all_y:
                                arr = np.asarray(all_y, dtype=float)
                                q1, q99 = np.percentile(arr, [1.0, 99.0])
                                if q1 == q99:
                                    q1 -= 0.5
                                    q99 += 0.5
                                span = q99 - q1
                                pad = 0.05 * span
                                row_y_limits_z[L] = (q1 - pad, q99 + pad)
                        axes_arr = np.array([[axesz[i]] for i in range(nrows_e)]) if ncols_e == 1 else axesz
                        L_labels = ['L=0 (s)', 'L=1 (p)', 'L=2 (d)', 'L=3 (f)', 'L=4 (g)']
                        for iL in range(5):
                            series = lscatter[iL].get(z)
                            cats_dict = series.get('cats') if series else None
                            for j, cat_lbl in enumerate(top_cats_z):
                                ax = axes_arr[iL, j]
                                if cats_dict and (cat_lbl in cats_dict) and cats_dict[cat_lbl]['x']:
                                    X = np.asarray(cats_dict[cat_lbl]['x'], dtype=float)
                                    Y = np.asarray(cats_dict[cat_lbl]['y'], dtype=float)
                                    perm = np.random.permutation(len(X))
                                    X = X[perm]
                                    Y = Y[perm]
                                    ax.scatter(X, Y, s=0.5, alpha=0.35, edgecolors='none', c='steelblue')
                                if iL in row_y_limits_z:
                                    ax.set_ylim(row_y_limits_z[iL])
                                else:
                                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=8, color='gray')
                                    if iL in row_y_limits_z:
                                        ax.set_ylim(row_y_limits_z[iL])
                                if iL == 0:
                                    ax.set_title(f'Neighbors: {cat_lbl}')
                                if j == 0:
                                    ax.set_ylabel(f"{L_labels[iL]}\nOverlap value")
                                if iL == nrows_e - 1:
                                    ax.set_xlabel('Normalized exponent α_new')
                        figz.suptitle(f'Z={z}: Overlap vs normalized exponent — rows=L, cols=neighbor types')
                        figz.tight_layout(rect=[0, 0.03, 1, 0.95])
                        figz.savefig(outdir / f'overlap_vs_norm_exponent_LxNeigh_Z{int(z)}.png', dpi=150)
                        plt.close(figz)
                except Exception as e:
                    print(f"[WARN] Failed generating per-element neighbor type grids: {e}")

                # Per-element figures: rows=L, cols=neighbor element presence (1,6,7,8)
                try:
                    elems2 = sorted({z for L in range(5) for z in lscatter[L].keys()})
                    col_elems_all = [1, 6, 7, 8]
                    L_labels = ['L=0 (s)', 'L=1 (p)', 'L=2 (d)', 'L=3 (f)', 'L=4 (g)']
                    for z in elems2:
                        ncols_e = len(col_elems_all)
                        nrows_e = 5
                        figc, axesc = plt.subplots(nrows_e, ncols_e, figsize=(3.5 * ncols_e, 3.0 * nrows_e), sharex=True, sharey=False)
                        row_y_limits_c = {}
                        for L in range(5):
                            seriesL = lscatter[L].get(z)
                            if not seriesL or not seriesL.get('cats'):
                                continue
                            all_y = []
                            for cat_lbl, cat_series in seriesL['cats'].items():
                                if cat_lbl == 'none':
                                    continue
                                ys = cat_series.get('y')
                                if ys:
                                    all_y.extend([yv for yv in ys if np.isfinite(yv)])
                            if all_y:
                                arr = np.asarray(all_y, dtype=float)
                                q1, q99 = np.percentile(arr, [1.0, 99.0])
                                if q1 == q99:
                                    q1 -= 0.5
                                    q99 += 0.5
                                span = q99 - q1
                                pad = 0.05 * span
                                row_y_limits_c[L] = (q1 - pad, q99 + pad)
                        for iL in range(5):
                            series = lscatter[iL].get(z)
                            cats_dict = series.get('cats') if series else None
                            for j, cz in enumerate(col_elems_all):
                                ax = axesc[iL, j]
                                X_all = []
                                Y_all = []
                                if cats_dict:
                                    for cat_lbl, cat_series in cats_dict.items():
                                        if not cat_series['x'] or cat_lbl == 'none':
                                            continue
                                        try:
                                            parts = [int(p) for p in cat_lbl.split('-') if p]
                                        except Exception:
                                            parts = []
                                        if cz in parts:
                                            X_all.extend(cat_series['x'])
                                            Y_all.extend(cat_series['y'])
                                if X_all:
                                    X = np.asarray(X_all, dtype=float)
                                    Y = np.asarray(Y_all, dtype=float)
                                    perm = np.random.permutation(len(X))
                                    X = X[perm]
                                    Y = Y[perm]
                                    ax.scatter(X, Y, s=0.5, alpha=0.35, edgecolors='none', c='steelblue')
                                if iL in row_y_limits_c:
                                    ax.set_ylim(row_y_limits_c[iL])
                                else:
                                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=8, color='gray')
                                    if iL in row_y_limits_c:
                                        ax.set_ylim(row_y_limits_c[iL])
                                if iL == 0:
                                    ax.set_title(f'Contains Z={cz}')
                                if j == 0:
                                    ax.set_ylabel(f"{L_labels[iL]}\nOverlap value")
                                if iL == nrows_e - 1:
                                    ax.set_xlabel('Normalized exponent α_new')
                        figc.suptitle(f'Z={z}: Overlap vs normalized exponent — rows=L, cols=neighbor element present')
                        figc.tight_layout(rect=[0, 0.03, 1, 0.95])
                        figc.savefig(outdir / f'overlap_vs_norm_exponent_LxNeighContains_Z{int(z)}.png', dpi=150)
                        plt.close(figc)
                except Exception as e:
                    print(f"[WARN] Failed generating per-element neighbor presence grids: {e}")
    except Exception:
        pass

    # ================= Aggregated L-only plots (all elements merged) =================
    if args.lscatter_aggregate:
        try:
            agg_bins = max(20, int(args.lscatter_aggregate_bins))
            agg_edges = np.linspace(0.0, 1.0, agg_bins + 1)
            agg_centers = 0.5 * (agg_edges[:-1] + agg_edges[1:])
            cap_total = int(args.lscatter_aggregate_cap)
            min_count_for_stats = 10
            # Containers per L
            agg_data = {L: {'x': [], 'y': []} for L in range(5)}
            agg_data_logz = {L: {'x': [], 'y': []} for L in range(5)}
            counts_total_points = [0]*5
            counts_total_points_logz = [0]*5
            # Gather all sampled series across elements & neighbor categories
            for L in range(5):
                for z, series in lscatter[L].items():
                    if not series or not series.get('cats'):
                        continue
                    for cat_series in series['cats'].values():
                        xs = cat_series.get('x')
                        ys = cat_series.get('y')
                        if not xs or not ys:
                            continue
                        if len(xs) != len(ys):
                            continue
                        agg_data[L]['x'].extend(xs)
                        agg_data[L]['y'].extend(ys)
                counts_total_points[L] = len(agg_data[L]['x'])
                # Downsample if exceeding cap
                if cap_total > 0 and counts_total_points[L] > cap_total:
                    idx = np.random.choice(counts_total_points[L], size=cap_total, replace=False)
                    x_arr = np.asarray(agg_data[L]['x'], dtype=float)[idx]
                    y_arr = np.asarray(agg_data[L]['y'], dtype=float)[idx]
                    agg_data[L]['x'] = x_arr.tolist()
                    agg_data[L]['y'] = y_arr.tolist()
                    counts_total_points[L] = cap_total
            # Gather log-z series
            for L in range(5):
                for z, series in lscatter_logz[L].items():
                    if not series or not series.get('cats'):
                        continue
                    for cat_series in series['cats'].values():
                        xs = cat_series.get('x'); ys = cat_series.get('y')
                        if not xs or not ys or len(xs) != len(ys):
                            continue
                        agg_data_logz[L]['x'].extend(xs)
                        agg_data_logz[L]['y'].extend(ys)
                counts_total_points_logz[L] = len(agg_data_logz[L]['x'])
                if cap_total > 0 and counts_total_points_logz[L] > cap_total:
                    idx2 = np.random.choice(counts_total_points_logz[L], size=cap_total, replace=False)
                    x_arr2 = np.asarray(agg_data_logz[L]['x'], dtype=float)[idx2]
                    y_arr2 = np.asarray(agg_data_logz[L]['y'], dtype=float)[idx2]
                    agg_data_logz[L]['x'] = x_arr2.tolist(); agg_data_logz[L]['y'] = y_arr2.tolist()
                    counts_total_points_logz[L] = cap_total
            # Compute per-L robust y-limits (1st-99th percentile + padding)
            agg_y_limits = {}
            for L in range(5):
                if counts_total_points[L] == 0:
                    continue
                arr_y = np.asarray(agg_data[L]['y'], dtype=float)
                arr_y = arr_y[np.isfinite(arr_y)]
                if arr_y.size == 0:
                    continue
                q1, q99 = np.percentile(arr_y, [1.0, 99.0])
                if not np.isfinite(q1) or not np.isfinite(q99):
                    continue
                if q1 == q99:
                    q1 -= 0.5
                    q99 += 0.5
                span = q99 - q1
                pad = 0.05 * span
                agg_y_limits[L] = (q1 - pad, q99 + pad)
            # Compute binned mean/std overlays per L
            agg_stats = {}
            for L in range(5):
                xL = np.asarray(agg_data[L]['x'], dtype=float)
                yL = np.asarray(agg_data[L]['y'], dtype=float)
                if xL.size == 0 or yL.size != xL.size:
                    continue
                idx = np.digitize(xL, agg_edges) - 1
                valid = (idx >= 0) & (idx < agg_bins)
                if not np.any(valid):
                    continue
                idx = idx[valid]
                yv = yL[valid]
                sum_y = np.bincount(idx, weights=yv, minlength=agg_bins)
                sum_y2 = np.bincount(idx, weights=yv*yv, minlength=agg_bins)
                counts = np.bincount(idx, minlength=agg_bins)
                with np.errstate(divide='ignore', invalid='ignore'):
                    mean_line = sum_y / counts
                    var_line = (sum_y2 / counts) - mean_line**2
                mean_line[counts < min_count_for_stats] = np.nan
                var_line[counts < min_count_for_stats] = np.nan
                std_line = np.sqrt(np.clip(var_line, 0.0, None))
                agg_stats[L] = {
                    'mean': mean_line,
                    'std': std_line,
                    'counts': counts,
                }
            # Plot unnormalized aggregated scatter with overlays
            figA, axesA = plt.subplots(1, 5, figsize=(22, 4.0), sharex=True, sharey=False)
            L_labels = ['L=0 (s)', 'L=1 (p)', 'L=2 (d)', 'L=3 (f)', 'L=4 (g)']
            for L in range(5):
                ax = axesA[L]
                xL = agg_data[L]['x']
                yL = agg_data[L]['y']
                if xL:
                    x_arr = np.asarray(xL, dtype=float)
                    y_arr = np.asarray(yL, dtype=float)
                    perm = np.random.permutation(len(x_arr))
                    x_arr = x_arr[perm]
                    y_arr = y_arr[perm]
                    ax.scatter(x_arr, y_arr, s=0.3, alpha=0.75, edgecolors='none', c='steelblue')
                if L in agg_stats:
                    st = agg_stats[L]
                    mean_line = st['mean']
                    std_line = st['std']
                    ax.plot(agg_centers, mean_line, color='black', linewidth=1.0)
                    ax.fill_between(agg_centers, mean_line - std_line, mean_line + std_line, color='black', alpha=0.12, linewidth=0)
                if L in agg_y_limits:
                    ax.set_ylim(agg_y_limits[L])
                ax.set_title(f"{L_labels[L]}\nN={counts_total_points[L]}")
                if L == 0:
                    ax.set_ylabel('Overlap value')
                ax.set_xlabel('Normalized exponent α_new')
            figA.suptitle('Aggregated overlap vs normalized exponent (all elements merged per L)')
            figA.tight_layout(rect=[0, 0.03, 1, 0.93])
            figA.savefig(outdir / 'overlap_vs_norm_exponent_aggregated_by_L.png', dpi=150)
            plt.close(figA)
            # Log-z aggregated unnormalized
            agg_edges_logz = np.linspace(-4.0, 4.0, agg_bins + 1)
            agg_centers_logz = 0.5 * (agg_edges_logz[:-1] + agg_edges_logz[1:])
            agg_stats_logz = {}
            for L in range(5):
                xL = np.asarray(agg_data_logz[L]['x'], dtype=float)
                yL = np.asarray(agg_data_logz[L]['y'], dtype=float)
                if xL.size == 0 or xL.size != yL.size:
                    continue
                idx = np.digitize(xL, agg_edges_logz) - 1
                valid = (idx >= 0) & (idx < agg_edges_logz.size - 1)
                if not np.any(valid):
                    continue
                idx = idx[valid]; yv = yL[valid]
                nb = agg_edges_logz.size - 1
                sum_y = np.bincount(idx, weights=yv, minlength=nb)
                sum_y2 = np.bincount(idx, weights=yv*yv, minlength=nb)
                counts = np.bincount(idx, minlength=nb)
                with np.errstate(divide='ignore', invalid='ignore'):
                    mean_line = sum_y / counts
                    var_line = (sum_y2 / counts) - mean_line**2
                mean_line[counts < min_count_for_stats] = np.nan
                var_line[counts < min_count_for_stats] = np.nan
                std_line = np.sqrt(np.clip(var_line, 0.0, None))
                agg_stats_logz[L] = {'mean': mean_line, 'std': std_line, 'counts': counts}
            figALZ, axesALZ = plt.subplots(1,5, figsize=(22,4.0), sharex=True, sharey=False)
            Y_CLIP = 10.0
            total_clip = 0
            for L in range(5):
                ax = axesALZ[L]
                xL = agg_data_logz[L]['x']; yL = agg_data_logz[L]['y']
                clipped_here = 0
                if xL:
                    x_arr = np.asarray(xL, dtype=float); y_arr = np.asarray(yL, dtype=float)
                    perm = np.random.permutation(len(x_arr)); x_arr = x_arr[perm]; y_arr = y_arr[perm]
                    outside = (y_arr > Y_CLIP) | (y_arr < -Y_CLIP)
                    clipped_here = int(np.sum(outside))
                    if clipped_here > 0:
                        total_clip += clipped_here
                        y_arr = np.clip(y_arr, -Y_CLIP, Y_CLIP)
                    ax.scatter(x_arr, y_arr, s=0.3, alpha=0.75, edgecolors='none', c='steelblue')
                if L in agg_stats_logz:
                    st = agg_stats_logz[L]
                    mean_line = np.clip(st['mean'], -Y_CLIP, Y_CLIP)
                    std_line = st['std']
                    ax.plot(agg_centers_logz, mean_line, color='black', linewidth=1.0)
                    # When filling, also clip
                    upper = np.clip(mean_line + std_line, -Y_CLIP, Y_CLIP)
                    lower = np.clip(mean_line - std_line, -Y_CLIP, Y_CLIP)
                    ax.fill_between(agg_centers_logz, lower, upper, color='black', alpha=0.12, linewidth=0)
                ax.set_ylim((-Y_CLIP, Y_CLIP))
                title_main = f"{L_labels[L]}\nN={counts_total_points_logz[L]}"
                if clipped_here > 0:
                    title_main += f"\nclipped {clipped_here}"
                ax.set_title(title_main, fontsize=9)
                if L == 0:
                    ax.set_ylabel(f'Overlap value (clipped ±{Y_CLIP})')
                ax.set_xlabel('Log-z normalized exponent')
            t_extra = '' if total_clip == 0 else f'  (total clipped points hidden: {total_clip})'
            figALZ.suptitle('Aggregated overlap vs log-z exponent (all elements merged per L)' + t_extra)
            figALZ.tight_layout(rect=[0,0.03,1,0.93])
            figALZ.savefig(outdir / 'overlap_vs_logz_exponent_aggregated_by_L.png', dpi=150)
            plt.close(figALZ)
            # Local z normalization over aggregated bins
            agg_z_data = {L: {'x': [], 'z': []} for L in range(5)}
            z_row_limits = {}
            for L in range(5):
                if L not in agg_stats:
                    continue
                st = agg_stats[L]
                mean_line = st['mean']
                std_line = st['std']
                counts = st['counts']
                xL = np.asarray(agg_data[L]['x'], dtype=float)
                yL = np.asarray(agg_data[L]['y'], dtype=float)
                if xL.size == 0 or yL.size != xL.size:
                    continue
                b_idx = np.digitize(xL, agg_edges) - 1
                valid = (b_idx >= 0) & (b_idx < agg_bins)
                if not np.any(valid):
                    continue
                b_idx = b_idx[valid]
                xL = xL[valid]
                yL = yL[valid]
                m = mean_line[b_idx]
                s = std_line[b_idx]
                cts = counts[b_idx]
                good = np.isfinite(m) & np.isfinite(s) & (s > 0) & (cts >= min_count_for_stats)
                if not np.any(good):
                    continue
                x_good = xL[good]
                y_good = yL[good]
                z_vals = (y_good - m[good]) / s[good]
                agg_z_data[L]['x'].extend(x_good.tolist())
                agg_z_data[L]['z'].extend(z_vals.tolist())
            # Robust z-limits
            for L in range(5):
                zL = agg_z_data[L]['z']
                if not zL:
                    continue
                arrz = np.asarray([v for v in zL if np.isfinite(v)], dtype=float)
                if arrz.size == 0:
                    continue
                q1, q99 = np.percentile(arrz, [1.0, 99.0])
                if q1 == q99:
                    q1 -= 0.5
                    q99 += 0.5
                span = q99 - q1
                pad = 0.05 * span
                z_row_limits[L] = (q1 - pad, q99 + pad)
            figZ, axesZ = plt.subplots(1, 5, figsize=(22, 4.0), sharex=True, sharey=False)
            for L in range(5):
                ax = axesZ[L]
                xL = agg_z_data[L]['x']
                zL = agg_z_data[L]['z']
                if xL:
                    x_arr = np.asarray(xL, dtype=float)
                    z_arr = np.asarray(zL, dtype=float)
                    perm = np.random.permutation(len(x_arr))
                    x_arr = x_arr[perm]
                    z_arr = z_arr[perm]
                    ax.scatter(x_arr, z_arr, s=0.3, alpha=0.6, edgecolors='none', c='steelblue')
                ax.axhline(0.0, color='black', linewidth=0.7, alpha=0.7)
                if L in z_row_limits:
                    ax.set_ylim(z_row_limits[L])
                ax.set_title(f"{L_labels[L]}")
                if L == 0:
                    ax.set_ylabel('Aggregated local z')
                ax.set_xlabel('Normalized exponent α_new')
            figZ.suptitle('Aggregated local z = (O - μ_bin)/σ_bin vs normalized exponent (all elements merged per L)')
            figZ.tight_layout(rect=[0, 0.03, 1, 0.93])
            figZ.savefig(outdir / 'overlap_localZ_vs_norm_exponent_aggregated_by_L.png', dpi=150)
            plt.close(figZ)
            # Aggregated log-z local z
            try:
                agg_z_data_logz = {L: {'x': [], 'z': []} for L in range(5)}
                for L in range(5):
                    if L not in agg_stats_logz:
                        continue
                    st = agg_stats_logz[L]
                    mean_line = st['mean']; std_line = st['std']; counts = st['counts']
                    xL = np.asarray(agg_data_logz[L]['x'], dtype=float)
                    yL = np.asarray(agg_data_logz[L]['y'], dtype=float)
                    if xL.size == 0 or xL.size != yL.size:
                        continue
                    b_idx = np.digitize(xL, agg_edges_logz) - 1
                    valid = (b_idx >= 0) & (b_idx < agg_edges_logz.size - 1)
                    if not np.any(valid):
                        continue
                    b_idx = b_idx[valid]; xL = xL[valid]; yL = yL[valid]
                    m = mean_line[b_idx]; s = std_line[b_idx]; cts = counts[b_idx]
                    good = np.isfinite(m) & np.isfinite(s) & (s > 0) & (cts >= min_count_for_stats)
                    if not np.any(good):
                        continue
                    zvals = (yL[good] - m[good]) / s[good]
                    agg_z_data_logz[L]['x'].extend(xL[good].tolist())
                    agg_z_data_logz[L]['z'].extend(zvals.tolist())
                figALZz, axesALZz = plt.subplots(1,5, figsize=(22,4.0), sharex=True, sharey=False)
                Y_CLIP = 10.0
                total_clip_z = 0
                for L in range(5):
                    ax = axesALZz[L]
                    xL = agg_z_data_logz[L]['x']; zL = agg_z_data_logz[L]['z']
                    clipped_here = 0
                    if xL:
                        x_arr = np.asarray(xL, dtype=float); z_arr = np.asarray(zL, dtype=float)
                        perm = np.random.permutation(len(x_arr)); x_arr = x_arr[perm]; z_arr = z_arr[perm]
                        outside = (z_arr > Y_CLIP) | (z_arr < -Y_CLIP)
                        clipped_here = int(np.sum(outside))
                        if clipped_here > 0:
                            total_clip_z += clipped_here
                            z_arr = np.clip(z_arr, -Y_CLIP, Y_CLIP)
                        ax.scatter(x_arr, z_arr, s=0.3, alpha=0.6, edgecolors='none', c='steelblue')
                    ax.axhline(0.0, color='black', linewidth=0.7, alpha=0.7)
                    ax.set_ylim((-Y_CLIP, Y_CLIP))
                    title_lbl = f"{L_labels[L]}"
                    if clipped_here > 0:
                        title_lbl += f"\nclipped {clipped_here}"
                    ax.set_title(title_lbl, fontsize=9)
                    if L == 0:
                        ax.set_ylabel(f'Aggregated local z (clipped ±{Y_CLIP})')
                    ax.set_xlabel('Log-z normalized exponent')
                t_extra = '' if total_clip_z == 0 else f'  (total clipped points hidden: {total_clip_z})'
                figALZz.suptitle('Aggregated local z vs log-z exponent (all elements merged per L)' + t_extra)
                figALZz.tight_layout(rect=[0,0.03,1,0.93])
                figALZz.savefig(outdir / 'overlap_localZ_vs_logz_exponent_aggregated_by_L.png', dpi=150)
                plt.close(figALZz)
            except Exception as e:
                print(f"[WARN] Failed aggregated log-z local z plots: {e}")
        except Exception as e:
            print(f"[WARN] Failed generating aggregated L-only plots: {e}")

    # ================= Curve-based normalized aggregated L plots =================
    if args.curve_norm_plots and curve_payload is not None:
        try:
            # Convert lists to arrays
            curve_data = {}
            for L in range(5):
                xs = np.asarray(curve_aggr[L]['x'], dtype=np.float64)
                zs = np.asarray(curve_aggr[L]['z'], dtype=np.float64)
                mask = np.isfinite(xs) & np.isfinite(zs)
                xs = xs[mask]
                zs = zs[mask]
                curve_data[L] = {'x': xs, 'z': zs}
            # Diagnostic: show distribution of x values (rounded) to confirm variation
            try:
                all_x_diag = np.concatenate([curve_data[L]['x'] for L in range(5)])
                if all_x_diag.size > 0:
                    uniq_x = np.unique(np.round(all_x_diag, 6))
                    if uniq_x.size <= 40:
                        print(f"[curve-norm] Unique normalized exponent values (rounded 6dp): {uniq_x.tolist()}")
                    else:
                        print(f"[curve-norm] {uniq_x.size} unique normalized exponent values (rounded 6dp). First10={uniq_x[:10].tolist()} Last10={uniq_x[-10:].tolist()}")
            except Exception:
                pass
            # Determine y limits per L (robust) with clamp to ±ymax
            ymax = float(args.curve_norm_ymax)
            y_limits = {}
            for L in range(5):
                z = curve_data[L]['z']
                if z.size == 0:
                    y_limits[L] = (-1.0, 1.0)
                    continue
                lo, hi = np.percentile(z, [1, 99]) if z.size > 100 else (z.min(), z.max())
                lo = max(lo - 0.1 * (hi - lo), -ymax)
                hi = min(hi + 0.1 * (hi - lo), ymax)
                lo = float(max(lo, -ymax))
                hi = float(min(hi, ymax))
                if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
                    lo, hi = -ymax, ymax
                y_limits[L] = (lo, hi)

            # Stats overlays (binned mean and ±std)
            bins = max(20, int(args.curve_norm_bins))
            edges = np.linspace(0.0, 1.0, bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            stats = {}
            for L in range(5):
                xs = curve_data[L]['x']
                zs = curve_data[L]['z']
                if xs.size == 0:
                    stats[L] = {'mean': np.full(bins, np.nan), 'std': np.full(bins, np.nan), 'count': np.zeros(bins, dtype=int)}
                    continue
                idx = np.digitize(xs, edges) - 1
                idx = np.clip(idx, 0, bins - 1)
                mean_b = np.full(bins, np.nan)
                std_b = np.full(bins, np.nan)
                cnt_b = np.zeros(bins, dtype=int)
                for b in range(bins):
                    m = idx == b
                    if np.any(m):
                        vv = zs[m]
                        mean_b[b] = float(np.mean(vv))
                        std_b[b] = float(np.std(vv))
                        cnt_b[b] = int(np.sum(m))
                stats[L] = {'mean': mean_b, 'std': std_b, 'count': cnt_b}

            # Plot
            fig, axes = plt.subplots(1, 5, figsize=(22, 4.0), sharex=True, sharey=False)
            L_labels = ['L=0 (s)', 'L=1 (p)', 'L=2 (d)', 'L=3 (f)', 'L=4 (g)']
            for L in range(5):
                ax = axes[L]
                xs = curve_data[L]['x']
                zs = np.clip(curve_data[L]['z'], -ymax, ymax)
                ax.scatter(xs, zs, s=3, alpha=0.15, edgecolor='none', color='tab:blue')
                # Overlays
                m = stats[L]['mean']
                s = stats[L]['std']
                ax.plot(centers, m, color='black', linewidth=1.6, zorder=5, label='Mean (binned)')
                ax.plot(centers, m + s, color='red', linewidth=1.0, linestyle='--', zorder=5, label='+1σ')
                ax.plot(centers, m - s, color='red', linewidth=1.0, linestyle='--', zorder=5, label='-1σ')
            # Annotate meta min/max comparison in first subplot
            try:
                if 'alpha_min' in curve_payload and 'alpha_max' in curve_payload and global_min is not None and global_max is not None:
                    txt = (f"curve α∈[{curve_payload['alpha_min']:.2g},{curve_payload['alpha_max']:.2g}]\n"
                           f"data α∈[{global_min:.2g},{global_max:.2g}]\n"
                           f"mode={'global' if args.curve_use_global_minmax else 'curve'}")
                    axes[0].text(0.02, 0.98, txt, transform=axes[0].transAxes, va='top', ha='left', fontsize=8,
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
            except Exception:
                pass
                ax.set_title(L_labels[L])
                ax.set_ylim(*y_limits[L])
                ax.grid(alpha=0.2)
                ax.set_xlabel('Normalized exponent x')
                if L == 0:
                    ax.set_ylabel('Curve-normalized z')
            # Deduplicate legend entries
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc='upper right', frameon=True)
            fig.suptitle('Curve-normalized overlap z = (O - μ_L(x))/σ_L(x) vs normalized exponent (aggregated by L)')
            fig.tight_layout(rect=[0, 0.03, 1, 0.93])
            fig.savefig(outdir / 'overlap_curveZ_vs_norm_exponent_aggregated_by_L.png', dpi=150)
            plt.close(fig)
            print("Generated curve-normalized L plots.")
            # Per-L z histograms
            try:
                hist_bins = max(20, int(args.curve_norm_hist_bins))
                figH, axesH = plt.subplots(1, 5, figsize=(22, 3.8), sharey=True)
                labels_L = ['L=0 (s)', 'L=1 (p)', 'L=2 (d)', 'L=3 (f)', 'L=4 (g)']
                for L in range(5):
                    ax = axesH[L]
                    z = curve_data[L]['z']
                    if z.size == 0:
                        ax.set_title(labels_L[L])
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=9)
                        continue
                    z_abs = np.abs(z)
                    # Robust symmetric range
                    if z_abs.size > 500:
                        z_hi = np.percentile(z_abs, 99.5)
                    else:
                        z_hi = z_abs.max()
                    z_hi = min(z_hi * 1.05, float(args.curve_norm_ymax))
                    if not np.isfinite(z_hi) or z_hi <= 0:
                        z_hi = float(args.curve_norm_ymax)
                    edges = np.linspace(-z_hi, z_hi, hist_bins + 1)
                    counts, _ = np.histogram(np.clip(z, -z_hi, z_hi), bins=edges)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    width = (edges[1] - edges[0]) * 0.999
                    # Avoid log transform; show raw counts; optionally annotate total
                    ax.bar(centers, np.log(counts + 1), width=width, color='tab:blue', alpha=0.85)
                    ax.set_title(labels_L[L])
                    if L == 0:
                        ax.set_ylabel('log(Count)')
                    ax.set_xlabel('Curve-normalized z')
                    ax.grid(alpha=0.2)
                figH.suptitle('Curve-normalized overlap z histograms by L')
                figH.tight_layout(rect=[0,0.05,1,0.95])
                figH.savefig(outdir / 'overlap_curveZ_histograms_by_L.png', dpi=150)
                plt.close(figH)
                print("Generated curve-normalized per-L z histograms.")
            except Exception as e_hist:
                print(f"[WARN] Failed generating curve-normalized histograms: {e_hist}")

            # ============ Curve-based local Z per L (aggregated) ============
            try:
                # We recompute z locally per sample using curve μ/σ at its x (defensive in case stored z had any earlier jitter side-effects)
                x_grid = curve_payload['x_grid']
                mean_grid = curve_payload['mean_grid']
                std_grid = curve_payload['std_grid']
                local_curve_data = {L: {'x': [], 'z': []} for L in range(5)}
                for L in range(5):
                    xs = curve_data[L]['x']
                    # Need original (O) not just pre-normalized z to recompute; we only stored z earlier.
                    # Since we don't retain raw O in curve_aggr, fallback: reuse stored z (already (O-μ)/σ).
                    # For fidelity, if future raw O is needed, extend reservoir to keep it. Here: treat existing z as final.
                    # We'll still build statistical overlays across x bins.
                    local_curve_data[L]['x'] = xs
                    local_curve_data[L]['z'] = curve_data[L]['z']
                # Binning for overlays
                bins_loc = max(40, int(args.curve_norm_bins))
                edges_loc = np.linspace(0.0, 1.0, bins_loc + 1)
                centers_loc = 0.5 * (edges_loc[:-1] + edges_loc[1:])
                stats_loc = {}
                for L in range(5):
                    xs = np.asarray(local_curve_data[L]['x'], dtype=float)
                    zvals = np.asarray(local_curve_data[L]['z'], dtype=float)
                    if xs.size == 0 or zvals.size != xs.size:
                        stats_loc[L] = {'mean': np.full(bins_loc, np.nan), 'std': np.full(bins_loc, np.nan), 'cnt': np.zeros(bins_loc, dtype=int)}
                        continue
                    idx = np.digitize(xs, edges_loc) - 1
                    idx = np.clip(idx, 0, bins_loc - 1)
                    mean_b = np.full(bins_loc, np.nan)
                    std_b = np.full(bins_loc, np.nan)
                    cnt_b = np.zeros(bins_loc, dtype=int)
                    for b in range(bins_loc):
                        m = idx == b
                        if np.any(m):
                            vv = zvals[m]
                            mean_b[b] = float(np.mean(vv))
                            std_b[b] = float(np.std(vv))
                            cnt_b[b] = int(np.sum(m))
                    stats_loc[L] = {'mean': mean_b, 'std': std_b, 'cnt': cnt_b}
                # Robust y-limits per L
                y_limits_loc = {}
                for L in range(5):
                    zv = np.asarray(local_curve_data[L]['z'], dtype=float)
                    zv = zv[np.isfinite(zv)]
                    if zv.size == 0:
                        y_limits_loc[L] = (-1, 1)
                        continue
                    lo, hi = np.percentile(zv, [1.0, 99.0]) if zv.size > 200 else (zv.min(), zv.max())
                    if lo == hi:
                        lo -= 0.5; hi += 0.5
                    span = hi - lo
                    pad = 0.05 * span
                    lo -= pad; hi += pad
                    ymax_clip = float(args.curve_norm_ymax)
                    lo = max(lo, -ymax_clip)
                    hi = min(hi, ymax_clip)
                    y_limits_loc[L] = (lo, hi)
                figCL, axesCL = plt.subplots(1, 5, figsize=(22, 4.0), sharex=True, sharey=False)
                L_labels = ['L=0 (s)', 'L=1 (p)', 'L=2 (d)', 'L=3 (f)', 'L=4 (g)']
                for L in range(5):
                    ax = axesCL[L]
                    xs = np.asarray(local_curve_data[L]['x'], dtype=float)
                    zv = np.asarray(local_curve_data[L]['z'], dtype=float)
                    if xs.size > 0:
                        ax.scatter(xs, np.clip(zv, -args.curve_norm_ymax, args.curve_norm_ymax), s=2, alpha=0.25, edgecolor='none', color='tab:blue')
                    st = stats_loc[L]
                    ax.plot(centers_loc, st['mean'], color='black', linewidth=1.3, zorder=5, label='Mean (binned)')
                    ax.plot(centers_loc, st['mean'] + st['std'], color='red', linestyle='--', linewidth=0.9, zorder=5, label='+1σ')
                    ax.plot(centers_loc, st['mean'] - st['std'], color='red', linestyle='--', linewidth=0.9, zorder=5, label='-1σ')
                    ax.set_title(L_labels[L])
                    ax.set_ylim(*y_limits_loc[L])
                    ax.grid(alpha=0.2)
                    ax.set_xlabel('Normalized exponent x')
                    if L == 0:
                        ax.set_ylabel('Curve-local z = (O-μ_L(x))/σ_L(x)')
                handlesCL, labelsCL = axesCL[0].get_legend_handles_labels()
                if handlesCL:
                    figCL.legend(handlesCL, labelsCL, loc='upper right', frameon=True)
                figCL.suptitle('Curve-based local z vs normalized exponent (aggregated by L)')
                figCL.tight_layout(rect=[0, 0.03, 1, 0.93])
                figCL.savefig(outdir / 'overlap_curveLocalZ_vs_norm_exponent_aggregated_by_L.png', dpi=150)
                plt.close(figCL)
                print('Generated curve-based local Z aggregated L plot.')
            except Exception as e:
                print(f"[WARN] Failed generating curve-based local Z aggregated plot: {e}")
        except Exception as e:
            print(f"[WARN] Failed generating curve-normalized L plots: {e}")

    # Save a small text summary
    summary_path = outdir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Loaded molecules: {meta_summary['loaded']}\n")
        f.write(f"Skipped molecules: {meta_summary['skipped']}\n")
        f.write(f"Global exponent min/max used: min={global_min:.6g}, max={global_max:.6g} (computed from {used_count} files)\n")
        if meta_summary['per_atom_basis_values']:
            f.write(f"Per-atom basis unique: {sorted(set(meta_summary['per_atom_basis_values']))}\n")
        if meta_summary['n_atoms_values']:
            f.write(f"n_atoms stats: min={min(meta_summary['n_atoms_values'])}, max={max(meta_summary['n_atoms_values'])}, "
                    f"median={int(np.median(meta_summary['n_atoms_values']))}\n")
        if meta_summary['E_values']:
            f.write(f"Exponents per molecule unique: {sorted(set(meta_summary['E_values']))}\n")
        # Simple correlation
        corr = np.corrcoef(norm_alpha_cat, mean_abs_cat)[0, 1]
        f.write(f"Pearson corr(norm_alpha, mean_abs_overlap): {corr:.4f}\n")
        if args.apply_overlap_normalization and overlap_mean_L is not None and overlap_std_L is not None:
            f.write("Applied per-L overlap normalization (O - mu_L)/sigma_L with stats:\n")
            for L in range(5):
                mu = overlap_mean_L[L] if L < len(overlap_mean_L) else float('nan')
                sd = overlap_std_L[L] if L < len(overlap_std_L) else float('nan')
                f.write(f"  L={L}: mean={mu:.6g}, std={sd:.6g}\n")
        if element_aggr:
            elems_list = sorted(element_aggr.keys())
            f.write(f"Heatmaps generated for elements: {elems_list}\n")
        if element_aggr_logz:
            f.write("Log-z heatmaps generated (duplicate set).\n")
        if element_dist:
            elems_list = sorted(element_dist.keys())
            f.write(f"Distribution boxplots generated for elements: {elems_list}\n")
        if 'exp_stats' in locals() and exp_stats:
            f.write("Exponent statistics (all sampled molecules):\n")
            f.write(f"  α:   min={exp_stats['exp_min']:.6g}, max={exp_stats['exp_max']:.6g}, mean={exp_stats['exp_mean']:.6g}, std={exp_stats['exp_std']:.6g}\n")
            f.write(f"  logα: min={exp_stats['log_min']:.6g}, max={exp_stats['log_max']:.6g}, mean={exp_stats['log_mean']:.6g}, std={exp_stats['log_std']:.6g}\n")
            f.write("Histogram figure: exponent_value_histograms.png\n")
        if args.curve_norm_plots and curve_payload is not None:
            f.write("Curve-based normalization plots generated: overlap_curveZ_vs_norm_exponent_aggregated_by_L.png\n")
            if (outdir / 'overlap_curveZ_histograms_by_L.png').exists():
                f.write("Curve-based per-L z histograms: overlap_curveZ_histograms_by_L.png\n")

    print(f"Saved plots and summary to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

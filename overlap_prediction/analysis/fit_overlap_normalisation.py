#!/usr/bin/env python3
"""
Fit smooth normalization functions for overlap integrals vs normalized exponents.

This script scans a dataset of CustomMolecule pickles and:
  1) Computes global min/max of exponent_values across molecules.
  2) Normalizes exponents: alpha_new = log(alpha/alpha_min) / log(alpha_max/alpha_min).
  3) For each L in {0..4}, aggregates per-atom, per-exponent overlap values into bins along alpha_new
     (default 1000 bins) to compute mean and std per bin of O(alpha, L).
  4) Fits smooth functions (splines) for mean_L(alpha_new) and std_L(alpha_new) using SciPy.
     For std, we fit in log domain to ensure positivity.
  5) Saves caches (JSON for alpha min/max and metadata, NPZ for binned arrays and sampled fits), and
     plots comparing binned statistics vs fitted curves, plus histograms of normalized overlaps.

Usage example:
  python scripts/fit_overlap_normalization.py \
      --data-dir /export/data/hmichael/scdp/data/full_comp_new \
      --outdir plots/overlap_norm_fit \
      --bins 1000 --sample-size 200

This script is standalone for experimentation. Later, the fitted functions can be wired into
the model transforms for on-the-fly normalization of overlap targets.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Optional torch import for tensor handling
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

# SciPy for spline fitting
from scipy.interpolate import UnivariateSpline, PchipInterpolator, LSQUnivariateSpline
from scipy.signal import savgol_filter


# Ensure repo root is on sys.path when running directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from overlap_pred.custom_data_2 import CustomMolecule  # type: ignore


# First 25 positions per-atom correspond to L=0..4 with these spans (start inclusive, end exclusive)
L_SPANS: List[Tuple[int, int]] = [(0, 1), (1, 4), (4, 9), (9, 16), (16, 25)]


def progress(seq, desc: str):
    if tqdm is None:
        return seq
    return tqdm(seq, desc=desc, mininterval=0.5)


def try_load_molecule(pkl_path: Path) -> CustomMolecule | None:
    """Best-effort loader for CustomMolecule, with support for compressed variants.

    Returns None on failure or invalid content.
    """
    try:
        mol = CustomMolecule.load_pickle(pkl_path)
        if mol is None:
            return None
        if getattr(mol, "exponent_values", None) is None:
            return None
        if getattr(mol, "overlap_int_2d", None) is None:
            return None
        return mol
    except Exception:
        # Allow for older compressed pickle formats if present in codebase
        try:
            from overlap_pred.custom_data_2 import CompressedCustomMolecule  # type: ignore
        except Exception:
            CompressedCustomMolecule = None  # type: ignore
        if CompressedCustomMolecule is None:
            return None
        try:
            mol = CompressedCustomMolecule.load_pickle(pkl_path)  # type: ignore
            if mol is None:
                return None
            # Expand to CustomMolecule if API provides it
            if hasattr(mol, "to_custom"):
                mol = mol.to_custom()
            if getattr(mol, "exponent_values", None) is None:
                return None
            if getattr(mol, "overlap_int_2d", None) is None:
                return None
            return mol
        except Exception:
            return None


def compute_global_alpha_minmax(files: List[Path], sample_size: int | None = None) -> tuple[float, float, int]:
    """Compute global positive finite min/max of exponents across given files."""
    if sample_size is not None and sample_size > 0 and len(files) > sample_size:
        rng = random.Random(123)
        files_iter = rng.sample(files, sample_size)
    else:
        files_iter = files
    gmin = math.inf
    gmax = -math.inf
    used = 0
    for fp in files_iter:
        mol = try_load_molecule(fp)
        if mol is None or getattr(mol, "exponent_values", None) is None:
            continue
        exps = mol.exponent_values
        if torch is not None and hasattr(exps, "detach"):
            exps = exps.detach().cpu().numpy()
        exps = np.asarray(exps)
        exps = exps[np.isfinite(exps) & (exps > 0)]
        if exps.size == 0:
            continue
        local_min = float(np.min(exps))
        local_max = float(np.max(exps))
        if np.isfinite(local_min) and local_min > 0:
            gmin = min(gmin, local_min)
        if np.isfinite(local_max) and local_max > 0:
            gmax = max(gmax, local_max)
        used += 1
    if not (np.isfinite(gmin) and np.isfinite(gmax)):
        raise RuntimeError("No valid exponents found to compute min/max")
    return float(gmin), float(gmax), used


def normalize_exponents(alphas: np.ndarray, alpha_min: float, alpha_max: float) -> np.ndarray:
    """Normalize exponents to [0,1] via log spacing: log(a/a_min)/log(a_max/a_min)."""
    alphas = np.asarray(alphas, dtype=np.float64)
    if not (np.isfinite(alpha_min) and np.isfinite(alpha_max) and alpha_min > 0 and alpha_max > 0):
        return np.zeros_like(alphas)
    if math.isclose(alpha_min, alpha_max):
        return np.zeros_like(alphas)
    denom = math.log(alpha_max / alpha_min)
    if denom == 0.0:
        return np.zeros_like(alphas)
    alphas_c = np.clip(alphas, alpha_min, alpha_max)
    return np.log(alphas_c / alpha_min) / denom


def extract_perL_values(mol: CustomMolecule) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Return (exps[E], overlaps[E,M], n_atoms, per_atom_basis) as numpy arrays, or raise.

    Converts tensors to numpy if needed. Validates shapes and per-atom basis >= 25.
    """
    exps = mol.exponent_values
    ovs = mol.overlap_int_2d
    if torch is not None and hasattr(exps, "detach"):
        exps = exps.detach().cpu().numpy()
    if torch is not None and hasattr(ovs, "detach"):
        ovs = ovs.detach().cpu().numpy()
    exps = np.asarray(exps)
    ovs = np.asarray(ovs)
    if exps.ndim != 1 or ovs.ndim != 2 or ovs.shape[0] != exps.shape[0]:
        raise ValueError("Invalid shapes: exps (E,), overlaps (E,M)")
    # Determine n_atoms
    if getattr(mol, "atom_types", None) is not None:
        at = mol.atom_types
        if torch is not None and hasattr(at, "detach"):
            at = at.detach().cpu().numpy()
        n_atoms = int(len(at))
    else:
        n_atoms = -1
    if n_atoms <= 0:
        raise ValueError("Invalid n_atoms")
    M = ovs.shape[1]
    if M % n_atoms != 0:
        raise ValueError("Overlap columns not divisible by n_atoms")
    pab = M // n_atoms
    if pab < 25:
        raise ValueError("Per-atom basis < 25; cannot split L=0..4 spans")
    return exps, ovs, n_atoms, pab


def aggregate_binned_stats(
    files: List[Path],
    alpha_min: float,
    alpha_max: float,
    bins: int = 1000,
    sample_size: int | None = None,
    seed: int = 42,
    per_atom_mean_over_m: bool = True,
) -> Dict[str, np.ndarray]:
    """Aggregate per-L overlap statistics in bins of normalized exponent.

    Returns dict with keys:
        edges: (bins+1,)
        centers: (bins,)
        sum_L: (5,bins)
        sumsq_L: (5,bins)
        count_L: (5,bins)
    """
    if sample_size is not None and sample_size > 0 and len(files) > sample_size:
        rng = random.Random(seed)
        files_iter = rng.sample(files, sample_size)
    else:
        files_iter = files

    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sum_L = np.zeros((5, bins), dtype=np.float64)
    sumsq_L = np.zeros((5, bins), dtype=np.float64)
    count_L = np.zeros((5, bins), dtype=np.int64)

    for fp in progress(files_iter, desc="Binning overlaps"):
        mol = try_load_molecule(fp)
        if mol is None:
            continue
        try:
            exps, ovs, n_atoms, pab = extract_perL_values(mol)
        except Exception:
            continue
        # Compute normalized exponents per row
        norm = normalize_exponents(exps, alpha_min, alpha_max)
        # For each exponent row, build per-atom per-L scalar (mean over m positions or flatten)
        E, M = ovs.shape
        # reshape to [E, n_atoms, pab]
        arr = ovs.reshape(E, n_atoms, pab)
        for e in range(E):
            b = int(np.clip(np.digitize(norm[e], edges) - 1, 0, bins - 1))
            # skip NaN or out of range cases
            if not np.isfinite(norm[e]):
                continue
            for L, (s, t) in enumerate(L_SPANS):
                block = arr[e, :, s:t]  # [n_atoms, mL]
                if per_atom_mean_over_m:
                    vals = block.mean(axis=1)  # per atom scalar
                else:
                    vals = block.reshape(-1)  # flatten across atoms and m
                # accumulate
                v = vals[np.isfinite(vals)]
                if v.size == 0:
                    continue
                sum_L[L, b] += float(v.sum())
                sumsq_L[L, b] += float((v * v).sum())
                count_L[L, b] += int(v.size)

    return {
        "edges": edges,
        "centers": centers,
        "sum_L": sum_L,
        "sumsq_L": sumsq_L,
        "count_L": count_L,
    }


def compute_means_stds(aggr: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std arrays of shape (5, bins) from aggregation."""
    sum_L = aggr["sum_L"]
    sumsq_L = aggr["sumsq_L"]
    count_L = aggr["count_L"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.divide(sum_L, count_L, out=np.zeros_like(sum_L, dtype=np.float64), where=count_L > 0)
        var = np.divide(sumsq_L, count_L, out=np.zeros_like(sum_L, dtype=np.float64), where=count_L > 0) - mean**2
        var[var < 0] = 0.0
        std = np.sqrt(var)
    return mean, std


def fit_smooth_functions(
    centers: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    counts: np.ndarray | None = None,
    smooth: float = 1e-2,
    min_points: int = 8,
    mean_method: str = "weighted_spline",  # options: weighted_spline, spline, pchip, mlp, poly_ls, lsq_spline, pchip_smooth
    std_method: str = "log_spline",  # options: log_spline, pchip, pchip_smooth
    weight_gamma: float = 0.5,
    edge_blend_frac: float = 0.05,
    edge_blend_strength: float = 0.7,
    mean_savgol_frac: float = 0.02,
    # MLP hyperparameters
    mlp_hidden: int = 128,
    mlp_layers: int = 2,
    mlp_epochs: int = 500,
    mlp_lr: float = 5e-3,
    mlp_weight_decay: float = 1e-4,
    mlp_dropout: float = 0.0,
    mlp_val_frac: float = 0.15,
    mlp_patience: int = 50,
    mlp_activation: str = "tanh",
    # Polynomial LSQ hyperparameters
    mean_poly_degree: int = 7,
    mean_poly_ridge: float = 1e-3,
    # LSQ spline hyperparameters
    mean_lsq_num_knots: int = 12,
    # PCHIP + Savitzky–Golay smoothing hyperparameters
    mean_pchip_savgol_frac: float = 0.02,
    mean_pchip_savgol_polyorder: int = 3,
) -> Tuple[List, List]:
    """Fit smooth functions per L for mean and std.

    - mean: UnivariateSpline with smoothing factor 's' (controlled via smooth * N)
    - std: fit log(std+eps) then exp of spline to ensure positivity

    Returns two lists of callables f_mean[L](x in [0,1]) and f_std[L](x in [0,1]).
    """
    bins = centers.shape[0]
    s_factor = max(1.0, smooth * bins)
    f_mean = []
    f_std = []
    eps = 1e-12
    x = centers
    for L in range(5):
        # Mean fitting
        y_m_full = mean[L]
        mask_m = np.isfinite(y_m_full) & np.isfinite(x)
        x_m = x[mask_m]
        y_m = y_m_full[mask_m]
        # Optional light pre-smoothing to reduce noise prior to fitting
        if mean_savgol_frac and mean_savgol_frac > 0 and y_m.size >= 7:
            try:
                win = max(5, int(len(y_m) * mean_savgol_frac))
                if win % 2 == 0:
                    win += 1
                win = min(win, len(y_m) // 2 * 2 + 1)
                if win >= 5 and win <= len(y_m):
                    y_m = savgol_filter(y_m, window_length=win, polyorder=2, mode="interp")
            except Exception:
                pass
        # Build weights from counts if provided
        w_m = None
        if counts is not None:
            c_full = counts[L].astype(np.float64)
            c_m = c_full[mask_m]
            if c_m.size > 0:
                c_m = np.maximum(c_m, 1.0)
                w_m = (c_m / np.max(c_m)) ** float(weight_gamma)
        # Primary mean model
        if mask_m.sum() < min_points:
            if x_m.size >= 2:
                primary = PchipInterpolator(x_m, y_m, extrapolate=True)
            else:
                v = float(np.nan_to_num(y_m.mean() if y_m.size else 0.0))
                primary = lambda t, vv=v: np.full_like(np.asarray(t, dtype=float), vv, dtype=float)
        else:
            if mean_method == "pchip":
                primary = PchipInterpolator(x_m, y_m, extrapolate=True)
            elif mean_method == "lsq_spline":
                # Least-squares cubic spline with a limited number of interior knots
                try:
                    n_knots = max(1, int(mean_lsq_num_knots))
                    # Place interior knots at quantiles of x_m (exclude endpoints)
                    qs = np.linspace(0.0, 1.0, n_knots + 2)[1:-1]
                    t = np.quantile(x_m, qs)
                    # Ensure strictly inside the domain and unique
                    xmin, xmax = float(x_m.min()), float(x_m.max())
                    t = t[(t > xmin) & (t < xmax)]
                    t = np.unique(t)
                    if t.size == 0:
                        raise RuntimeError("No valid interior knots for LSQ spline")
                    primary = LSQUnivariateSpline(x_m, y_m, t, w=(w_m if w_m is not None else None), k=3, check_finite=False)
                except Exception:
                    primary = PchipInterpolator(x_m, y_m, extrapolate=True)
            elif mean_method == "pchip_smooth":
                # PCHIP followed by Savitzky–Golay smoothing on a dense grid, then re-interpolate
                try:
                    base = PchipInterpolator(x_m, y_m, extrapolate=True)
                    xx = np.linspace(float(x_m.min()), float(x_m.max()), 2049)
                    yy = np.asarray(base(xx))
                    # Apply Savitzky–Golay smoothing
                    frac = float(mean_pchip_savgol_frac)
                    polyorder = int(mean_pchip_savgol_polyorder)
                    if frac <= 0:
                        primary = base
                    else:
                        win = max(5, int(len(xx) * frac))
                        if win % 2 == 0:
                            win += 1
                        win = min(win, (len(xx)//2)*2 + 1)
                        if win < polyorder + 2:
                            win = polyorder + 3 if (polyorder + 3) % 2 == 1 else polyorder + 4
                        yys = savgol_filter(yy, window_length=win, polyorder=max(2, polyorder), mode="interp")
                        primary = PchipInterpolator(xx, yys, extrapolate=True)
                except Exception:
                    primary = PchipInterpolator(x_m, y_m, extrapolate=True)
            elif mean_method == "mlp":
                # MLP regressor (1D -> 1D)
                try:
                    if torch is None:
                        raise RuntimeError("PyTorch not available for mlp mean_method")
                    import torch.nn as nn
                    import torch.optim as optim

                    class MLP(nn.Module):
                        def __init__(self, hidden: int, layers: int, dropout: float, activation: str = "tanh"):
                            super().__init__()
                            acts = {
                                "tanh": nn.Tanh(),
                                "relu": nn.ReLU(),
                                "gelu": nn.GELU(),
                            }
                            act = acts.get(activation, nn.Tanh())
                            modules: List[nn.Module] = [nn.Linear(1, hidden), act]
                            if dropout and dropout > 0:
                                modules.append(nn.Dropout(dropout))
                            for _ in range(max(0, layers - 1)):
                                modules.extend([nn.Linear(hidden, hidden), act])
                                if dropout and dropout > 0:
                                    modules.append(nn.Dropout(dropout))
                            modules.append(nn.Linear(hidden, 1))
                            self.net = nn.Sequential(*modules)
                        def forward(self, x):
                            return self.net(x)

                    # Prepare tensors
                    X = torch.from_numpy(x_m.astype(np.float32)).view(-1, 1)
                    Y = torch.from_numpy(y_m.astype(np.float32)).view(-1, 1)
                    if w_m is not None:
                        W = torch.from_numpy(w_m.astype(np.float32)).view(-1, 1)
                        W = W / (W.mean() + 1e-8)
                    else:
                        W = torch.ones_like(Y)

                    # Train/val split
                    n = X.shape[0]
                    if n >= 10 and 0.0 < mlp_val_frac < 0.9:
                        idx = np.arange(n)
                        rng = np.random.RandomState(1234)
                        rng.shuffle(idx)
                        split = int(n * (1.0 - mlp_val_frac))
                        tr_idx = idx[:split] if split > 0 else idx
                        va_idx = idx[split:] if split < n else idx[:0]
                    else:
                        tr_idx = np.arange(n)
                        va_idx = np.arange(0)

                    Xtr, Ytr, Wtr = X[tr_idx], Y[tr_idx], W[tr_idx]
                    Xva, Yva, Wva = X[va_idx], Y[va_idx], W[va_idx]

                    model = MLP(mlp_hidden, mlp_layers, mlp_dropout, mlp_activation)
                    opt = optim.Adam(model.parameters(), lr=mlp_lr, weight_decay=mlp_weight_decay)

                    def loss_fn(pred, targ, w):
                        return ((w * (pred - targ) ** 2).mean())

                    best_state = None
                    best_val = float("inf")
                    no_improve = 0
                    model.train()
                    for epoch in range(int(mlp_epochs)):
                        opt.zero_grad()
                        pred = model(Xtr)
                        loss = loss_fn(pred, Ytr, Wtr)
                        loss.backward()
                        opt.step()
                        # validation
                        if Xva.numel() > 0:
                            model.eval()
                            with torch.no_grad():
                                vpred = model(Xva)
                                vloss = float(loss_fn(vpred, Yva, Wva).item())
                            model.train()
                        else:
                            vloss = float(loss.item())
                        if vloss < best_val - 1e-6:
                            best_val = vloss
                            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                            no_improve = 0
                        else:
                            no_improve += 1
                            if no_improve >= int(mlp_patience):
                                break
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    model.eval()
                    def primary(t, m=model):
                        with torch.no_grad():
                            tt = np.atleast_1d(np.asarray(t, dtype=np.float32)).reshape(-1, 1)
                            out = m(torch.from_numpy(tt)).cpu().numpy().reshape(-1)
                        return out
                except Exception:
                    # Fallback if anything goes wrong
                    primary = PchipInterpolator(x_m, y_m, extrapolate=True)
            elif mean_method == "poly_ls":
                # Standard weighted least squares polynomial fit with ridge regularization
                try:
                    deg = max(1, int(mean_poly_degree))
                    # Build Vandermonde in increasing powers for numerical stability with polynomial.polynomial
                    # X columns: [1, x, x^2, ... x^deg]
                    X = np.vander(x_m, N=deg + 1, increasing=True)
                    # Weights
                    if w_m is None:
                        w = np.ones_like(y_m)
                    else:
                        w = w_m
                    # Weighted ridge normal equations: (X^T W X + λI)β = X^T W y
                    W = np.diag(w.astype(np.float64))
                    XtW = X.T @ W
                    A = XtW @ X
                    # Ridge (do not regularize intercept too hard)
                    lam = float(mean_poly_ridge)
                    I = np.eye(A.shape[0])
                    I[0, 0] = 0.0  # don't penalize bias
                    A_reg = A + lam * I
                    b = XtW @ y_m
                    beta = np.linalg.solve(A_reg, b)
                    # Evaluator using polynomial in increasing order
                    def primary(t, b=beta):
                        tt = np.atleast_1d(np.asarray(t, dtype=float))
                        TT = np.vander(tt, N=b.shape[0], increasing=True)
                        out = TT @ b
                        return np.asarray(out).reshape(-1)
                except Exception:
                    primary = PchipInterpolator(x_m, y_m, extrapolate=True)
            else:
                try:
                    if mean_method == "weighted_spline" and w_m is not None:
                        primary = UnivariateSpline(x_m, y_m, w=w_m, s=max(1.0, smooth * len(x)))
                    else:
                        primary = UnivariateSpline(x_m, y_m, s=max(1.0, smooth * len(x)))
                except Exception:
                    primary = PchipInterpolator(x_m, y_m, extrapolate=True)
        # Edge blending with monotone PCHIP near boundaries to reduce overshoot
        try:
            edge_model = PchipInterpolator(x_m, y_m, extrapolate=True)
        except Exception:
            edge_model = primary
        f = float(edge_blend_frac)
        sstr = float(edge_blend_strength)
        def _blend_func_factory(p=primary, e=edge_model, f=f, sstr=sstr):
            def fn(t):
                tt = np.asarray(t, dtype=float)
                if f <= 0:
                    return np.asarray(p(tt))
                # weight w in [0, sstr] within edge fractions
                w = np.zeros_like(tt)
                left = tt < f
                right = tt > (1.0 - f)
                if np.any(left):
                    w[left] = sstr * (1.0 - tt[left] / f)
                if np.any(right):
                    w[right] = sstr * ((tt[right] - (1.0 - f)) / f)
                w = np.clip(w, 0.0, sstr)
                return (1.0 - w) * np.asarray(p(tt)) + w * np.asarray(e(tt))
            return fn
        f_mean.append(_blend_func_factory())

        # Std fitting (as before, with robust fallbacks)
        y_s_full = std[L]
        mask_s = np.isfinite(y_s_full) & np.isfinite(x) & (y_s_full >= 0)
        x_s = x[mask_s]
        y_s = y_s_full[mask_s]
        if mask_s.sum() < min_points:
            ss = np.clip(np.nan_to_num(y_s, nan=0.0), 0.0, None)
            try:
                win = max(5, (len(ss)//51)*2 + 5)
                if win % 2 == 0:
                    win += 1
                ss_s = savgol_filter(ss, window_length=min(win, len(ss)//2*2 + 1), polyorder=2)
            except Exception:
                ss_s = ss
            base_x = x_s if x_s.size > 1 else x
            base_y = ss_s if x_s.size > 1 else np.clip(np.nan_to_num(y_s_full, nan=0.0), 0.0, None)
            f_std.append(PchipInterpolator(base_x, np.maximum(base_y, 0.0), extrapolate=True))
        else:
            if std_method == "pchip":
                f_std.append(PchipInterpolator(x_s, np.maximum(y_s, 0.0), extrapolate=True))
            elif std_method == "pchip_smooth":
                try:
                    base = PchipInterpolator(x_s, np.maximum(y_s, 0.0), extrapolate=True)
                    xx = np.linspace(float(x_s.min()), float(x_s.max()), 2049)
                    yy = np.asarray(base(xx))
                    frac = float(mean_pchip_savgol_frac)
                    polyorder = int(mean_pchip_savgol_polyorder)
                    if frac <= 0:
                        f_std.append(base)
                    else:
                        win = max(5, int(len(xx) * frac))
                        if win % 2 == 0:
                            win += 1
                        win = min(win, (len(xx)//2)*2 + 1)
                        if win < polyorder + 2:
                            win = polyorder + 3 if (polyorder + 3) % 2 == 1 else polyorder + 4
                        yys = savgol_filter(yy, window_length=win, polyorder=max(2, polyorder), mode="interp")
                        yys = np.maximum(yys, 0.0)
                        f_std.append(PchipInterpolator(xx, yys, extrapolate=True))
                except Exception:
                    f_std.append(PchipInterpolator(x_s, np.maximum(y_s, 0.0), extrapolate=True))
            else:
                try:
                    ys = np.log(np.maximum(y_s, eps))
                    spl = UnivariateSpline(x_s, ys, s=max(1.0, smooth * len(x)))
                    f_std.append(lambda t, s=spl: np.exp(s(t)))
                except Exception:
                    f_std.append(PchipInterpolator(x_s, np.maximum(y_s, 0.0), extrapolate=True))
    return f_mean, f_std


def sample_fits(f_mean: List, f_std: List, num: int = 1001) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, num)
    Ymean = np.vstack([np.asarray(f(x)) for f in f_mean])
    Ystd = np.vstack([np.asarray(f(x)) for f in f_std])
    return x, Ymean, Ystd


def plot_perL_binned_vs_fit(
    outdir: Path,
    centers: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    f_mean: List,
    f_std: List,
    *,
    primary_mean_name: str = "fitted",
    compare_f_means: Dict[str, List] | None = None,
):
    outdir.mkdir(parents=True, exist_ok=True)
    x_grid, Ymean, Ystd = sample_fits(f_mean, f_std, num=1001)
    for L in range(5):
        # Mean plot
        plt.figure(figsize=(8, 5))
        plt.plot(centers, mean[L], label="binned mean", lw=1.5)
        plt.plot(x_grid, Ymean[L], label=f"mean ({primary_mean_name})", lw=2.3)
        # Overlay comparison mean fits if provided
        if compare_f_means:
            styles = ["--", "-.", ":", (0, (3, 1, 1, 1))]
            for i, (name, fms) in enumerate(compare_f_means.items()):
                try:
                    ym = np.asarray(fms[L](x_grid))
                    plt.plot(
                        x_grid,
                        ym,
                        linestyle=styles[i % len(styles)],
                        lw=1.8,
                        label=f"mean ({name})",
                    )
                except Exception:
                    # Skip plotting this method if it fails to evaluate
                    continue
        plt.xlabel("Normalized exponent α_new")
        plt.ylabel(f"Mean O (L={L})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"mean_fit_L{L}.png", dpi=150)
        plt.close()
        # Std plot
        plt.figure(figsize=(8, 5))
        plt.plot(centers, std[L], label="binned std", lw=1.5)
        plt.plot(x_grid, Ystd[L], label="fitted std", lw=2)
        plt.xlabel("Normalized exponent α_new")
        plt.ylabel(f"Std O (L={L})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"std_fit_L{L}.png", dpi=150)
        plt.close()


def demo_normalization_hist(
    outdir: Path,
    files: List[Path],
    alpha_min: float,
    alpha_max: float,
    f_mean: List,
    f_std: List,
    sample_mols: int = 20,
    seed: int = 42,
):
    """Quick histogram to visualize (O - mu)/sigma per L ~ N(0,1)."""
    rng = random.Random(seed)
    if len(files) > sample_mols:
        files_iter = rng.sample(files, sample_mols)
    else:
        files_iter = files
    vals_per_L = [[] for _ in range(5)]
    for fp in progress(files_iter, desc="Demo normalization"):
        mol = try_load_molecule(fp)
        if mol is None:
            continue
        try:
            exps, ovs, n_atoms, pab = extract_perL_values(mol)
        except Exception:
            continue
        norm = normalize_exponents(exps, alpha_min, alpha_max)
        E, M = ovs.shape
        arr = ovs.reshape(E, n_atoms, pab)
        for e in range(E):
            x = float(np.clip(norm[e], 0.0, 1.0))
            mu = np.array([f(x) for f in f_mean], dtype=np.float64).reshape(5)
            sd = np.array([f(x) for f in f_std], dtype=np.float64).reshape(5)
            sd = np.clip(sd, 1e-12, None)
            for L, (s, t) in enumerate(L_SPANS):
                block = arr[e, :, s:t]
                v = block.mean(axis=1)  # match aggregation choice
                z = (v - mu[L]) / sd[L]
                v_ok = z[np.isfinite(z)]
                if v_ok.size > 0:
                    vals_per_L[L].append(v_ok)
    # Plot
    outdir.mkdir(parents=True, exist_ok=True)
    for L in range(5):
        if len(vals_per_L[L]) == 0:
            continue
        v = np.concatenate(vals_per_L[L], axis=0)
        plt.figure(figsize=(7, 5))
        plt.hist(v, bins=100, density=True, alpha=0.7, color="steelblue")
        plt.title(f"Normalized O(alpha,L) histogram (L={L})")
        plt.xlabel("z = (O - mu_L(alpha))/sigma_L(alpha)")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"normalized_hist_L{L}.png", dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Fit per-L smooth normalization functions vs normalized exponents")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with molecule .pkl files")
    parser.add_argument("--outdir", type=str, default="plots/overlap_norm_fit", help="Output directory for caches and plots")
    parser.add_argument("--bins", type=int, default=1000, help="Number of bins along normalized exponent axis")
    parser.add_argument("--sample-size", type=int, default=0, help="Optional cap on number of molecules to process (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--minmax-scan-sample", type=int, default=0, help="If >0, estimate alpha min/max from this many random files; 0 = scan all")
    parser.add_argument("--recompute-minmax", action="store_true", help="Force recompute alpha min/max even if cached JSON exists")
    parser.add_argument("--recompute-fits", action="store_true", help="Force recompute fits even if cache exists")
    parser.add_argument("--smooth", type=float, default=1e-2, help="Relative smoothing strength for splines (scaled by number of bins)")
    parser.add_argument("--mean-method", type=str, default="pchip_smooth", choices=["weighted_spline", "spline", "pchip", "mlp", "poly_ls", "lsq_spline", "pchip_smooth"], help="Mean fit method")
    parser.add_argument("--std-method", type=str, default="pchip_smooth", choices=["log_spline", "pchip", "pchip_smooth"], help="Std fit method")
    parser.add_argument(
        "--compare-mean-methods",
        type=str,
        default="",
        help=(
            "Comma-separated list of additional mean fit methods to overlay on plots for comparison. "
            "Choices: weighted_spline,spline,pchip,mlp,poly_ls,lsq_spline,pchip_smooth"
        ),
    )
    parser.add_argument("--mean-weight-gamma", type=float, default=0.5, help="Exponent for count-based weights (0=no weighting, 1=linear)")
    parser.add_argument("--mean-edge-frac", type=float, default=0.2, help="Fraction of domain near edges to blend toward PCHIP")
    parser.add_argument("--mean-edge-strength", type=float, default=0.7, help="Blend strength toward PCHIP at edges (0..1)")
    parser.add_argument("--mean-savgol-frac", type=float, default=0.02, help="Savitzky–Golay pre-smoothing window as fraction of points for mean")
    # LSQ spline params
    parser.add_argument("--mean-lsq-num-knots", type=int, default=12, help="Number of interior knots for LSQ spline mean fitter (1D cubic)")
    # PCHIP+Savgol params
    parser.add_argument("--mean-pchip-savgol-frac", type=float, default=0.02, help="Fractional window length for Savitzky–Golay smoothing applied to PCHIP curve")
    parser.add_argument("--mean-pchip-savgol-polyorder", type=int, default=3, help="Savitzky–Golay polynomial order for PCHIP smoothing")
    # Polynomial LS mean fitter params
    parser.add_argument("--mean-poly-degree", type=int, default=7, help="Polynomial degree for least-squares mean fitter")
    parser.add_argument("--mean-poly-ridge", type=float, default=1e-3, help="Ridge regularization for polynomial LS mean fitter")
    # MLP mean fitter params
    parser.add_argument("--mean-mlp-hidden", type=int, default=128, help="MLP hidden size for mean fitter")
    parser.add_argument("--mean-mlp-layers", type=int, default=2, help="MLP hidden layers for mean fitter")
    parser.add_argument("--mean-mlp-epochs", type=int, default=500, help="MLP training epochs for mean fitter")
    parser.add_argument("--mean-mlp-lr", type=float, default=5e-3, help="MLP learning rate for mean fitter")
    parser.add_argument("--mean-mlp-wd", type=float, default=1e-4, help="MLP weight decay for mean fitter")
    parser.add_argument("--mean-mlp-dropout", type=float, default=0.0, help="MLP dropout for mean fitter")
    parser.add_argument("--mean-mlp-val-frac", type=float, default=0.15, help="Validation fraction for MLP mean fitter")
    parser.add_argument("--mean-mlp-patience", type=int, default=50, help="Early stopping patience for MLP mean fitter")
    parser.add_argument("--mean-mlp-activation", type=str, default="tanh", choices=["tanh","relu","gelu"], help="Activation function for MLP mean fitter")
    parser.add_argument("--flatten-per-atom", action="store_true", help="Instead of per-atom mean over m channels, flatten and use all values in L-span.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    assert data_dir.is_dir(), f"Data dir not found: {data_dir}"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files_all = sorted(list(data_dir.glob("*.pkl")))
    if not files_all:
        print(f"No .pkl files found in {data_dir}")
        return 1

    # Cache paths
    meta_json = outdir / "overlap_norm_meta.json"
    binned_npz = outdir / "overlap_binned_stats.npz"
    fit_npz = outdir / "overlap_fit_curves.npz"

    alpha_min = None
    alpha_max = None
    if meta_json.exists() and not args.recompute_minmax:
        try:
            with open(meta_json, "r") as f:
                meta = json.load(f)
            alpha_min = float(meta.get("alpha_min", np.nan))
            alpha_max = float(meta.get("alpha_max", np.nan))
            if not (np.isfinite(alpha_min) and np.isfinite(alpha_max)):
                alpha_min = None
                alpha_max = None
        except Exception:
            alpha_min = None
            alpha_max = None

    if alpha_min is None or alpha_max is None:
        print("Computing global alpha min/max ...")
        mm_sample = args.minmax_scan_sample if args.minmax_scan_sample and args.minmax_scan_sample > 0 else None
        alpha_min, alpha_max, used = compute_global_alpha_minmax(files_all, sample_size=mm_sample)
        with open(meta_json, "w") as f:
            json.dump({"alpha_min": alpha_min, "alpha_max": alpha_max, "used_files_minmax": used}, f, indent=2)
        print(f"alpha_min={alpha_min:.6g}, alpha_max={alpha_max:.6g} (from {used} files)")
    else:
        print(f"Loaded cached alpha_min/max: {alpha_min:.6g}, {alpha_max:.6g}")

    # Optionally subsample files for aggregation
    if args.sample_size and args.sample_size > 0 and len(files_all) > args.sample_size:
        rng = random.Random(args.seed)
        files = rng.sample(files_all, args.sample_size)
    else:
        files = files_all

    # Aggregate binned stats
    need_fit = True
    if binned_npz.exists() and not args.recompute_fits:
        try:
            npz = np.load(binned_npz)
            edges = npz["edges"]
            centers = npz["centers"]
            sum_L = npz["sum_L"]
            sumsq_L = npz["sumsq_L"]
            count_L = npz["count_L"]
            aggr = {"edges": edges, "centers": centers, "sum_L": sum_L, "sumsq_L": sumsq_L, "count_L": count_L}
            need_fit = True
        except Exception:
            aggr = None  # type: ignore
    else:
        aggr = None  # type: ignore

    if aggr is None:
        print("Aggregating binned per-L stats ...")
        aggr = aggregate_binned_stats(
            files=files,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            bins=int(args.bins),
            sample_size=None,
            seed=args.seed,
            per_atom_mean_over_m=not bool(args.flatten_per_atom),
        )
        np.savez_compressed(binned_npz, **aggr)
        print(f"Saved binned stats to {binned_npz}")

    centers = aggr["centers"]
    mean, std = compute_means_stds(aggr)

    # Fit smooth functions
    if fit_npz.exists() and not args.recompute_fits:
        try:
            fitz = np.load(fit_npz)
            x_grid = fitz["x_grid"]
            Ymean = fitz["mean_grid"]
            Ystd = fitz["std_grid"]
            # Build interpolators for evaluation in demo
            f_mean = [PchipInterpolator(x_grid, Ymean[L], extrapolate=True) for L in range(5)]
            f_std = [PchipInterpolator(x_grid, np.maximum(Ystd[L], 0.0), extrapolate=True) for L in range(5)]
            cached_fits = True
        except Exception:
            cached_fits = False
    else:
        cached_fits = False

    if not cached_fits:
        print("Fitting smooth functions ...")
        f_mean, f_std = fit_smooth_functions(
            centers,
            mean,
            std,
            counts=aggr.get("count_L"),
            smooth=float(args.smooth),
            mean_method=str(args.mean_method),
            std_method=str(args.std_method),
            weight_gamma=float(args.mean_weight_gamma),
            edge_blend_frac=float(args.mean_edge_frac),
            edge_blend_strength=float(args.mean_edge_strength),
            mean_savgol_frac=float(args.mean_savgol_frac),
            mean_poly_degree=int(args.mean_poly_degree),
            mean_poly_ridge=float(args.mean_poly_ridge),
            mlp_hidden=int(args.mean_mlp_hidden),
            mlp_layers=int(args.mean_mlp_layers),
            mlp_epochs=int(args.mean_mlp_epochs),
            mlp_lr=float(args.mean_mlp_lr),
            mlp_weight_decay=float(args.mean_mlp_wd),
            mlp_dropout=float(args.mean_mlp_dropout),
            mlp_val_frac=float(args.mean_mlp_val_frac),
            mlp_patience=int(args.mean_mlp_patience),
            mlp_activation=str(args.mean_mlp_activation),
            mean_lsq_num_knots=int(args.mean_lsq_num_knots),
            mean_pchip_savgol_frac=float(args.mean_pchip_savgol_frac),
            mean_pchip_savgol_polyorder=int(args.mean_pchip_savgol_polyorder),
        )
        x_grid, Ymean, Ystd = sample_fits(f_mean, f_std, num=2001)
        np.savez_compressed(fit_npz, x_grid=x_grid, mean_grid=Ymean, std_grid=Ystd)
        print(f"Saved fitted curves to {fit_npz}")

    # Optionally compute comparison mean fits (overlay on plots)
    compare_methods_raw = (args.compare_mean_methods or "").strip()
    compare_methods: List[str] = []
    valid_methods = {"weighted_spline", "spline", "pchip", "mlp", "poly_ls", "lsq_spline", "pchip_smooth"}
    if compare_methods_raw:
        compare_methods = [m.strip() for m in compare_methods_raw.split(",") if m.strip()]
        # Filter invalid and dedupe while preserving order
        seen = set()
        filtered: List[str] = []
        for m in compare_methods:
            if m not in valid_methods:
                print(f"[warn] Ignoring invalid compare method: {m}")
                continue
            if m in seen:
                continue
            seen.add(m)
            filtered.append(m)
        compare_methods = filtered
    # Compute compare fits
    compare_f_means: Dict[str, List] = {}
    for cm in compare_methods:
        if cm == args.mean_method:
            # We'll already plot the primary; keep it once.
            continue
        try:
            fm_c, _ = fit_smooth_functions(
                centers,
                mean,
                std,
                counts=aggr.get("count_L"),
                smooth=float(args.smooth),
                mean_method=str(cm),
                std_method=str(args.std_method),
                weight_gamma=float(args.mean_weight_gamma),
                edge_blend_frac=float(args.mean_edge_frac),
                edge_blend_strength=float(args.mean_edge_strength),
                mean_savgol_frac=float(args.mean_savgol_frac),
                mean_poly_degree=int(args.mean_poly_degree),
                mean_poly_ridge=float(args.mean_poly_ridge),
                mlp_hidden=int(args.mean_mlp_hidden),
                mlp_layers=int(args.mean_mlp_layers),
                mlp_epochs=int(args.mean_mlp_epochs),
                mlp_lr=float(args.mean_mlp_lr),
                mlp_weight_decay=float(args.mean_mlp_wd),
                mlp_dropout=float(args.mean_mlp_dropout),
                mlp_val_frac=float(args.mean_mlp_val_frac),
                mlp_patience=int(args.mean_mlp_patience),
                mlp_activation=str(args.mean_mlp_activation),
                mean_lsq_num_knots=int(args.mean_lsq_num_knots),
                mean_pchip_savgol_frac=float(args.mean_pchip_savgol_frac),
                mean_pchip_savgol_polyorder=int(args.mean_pchip_savgol_polyorder),
            )
            compare_f_means[cm] = fm_c
        except Exception as e:
            print(f"[warn] Failed to fit compare method '{cm}': {e}")

    # Plots
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_perL_binned_vs_fit(
        plot_dir,
        centers,
        mean,
        std,
        f_mean,
        f_std,
        primary_mean_name=str(args.mean_method),
        compare_f_means=compare_f_means or None,
    )
    demo_normalization_hist(plot_dir, files, alpha_min, alpha_max, f_mean, f_std, sample_mols=min(100, len(files)))

    # Save a small summary
    with open(outdir / "summary.txt", "w") as f:
        print(f"Files processed: {len(files)}", file=f)
        print(f"Bins: {int(args.bins)}", file=f)
        print(f"alpha_min: {alpha_min}", file=f)
        print(f"alpha_max: {alpha_max}", file=f)
        # quick stats on counts per L
        counts = aggr["count_L"]
        for L in range(5):
            print(f"L={L}: total samples {int(counts[L].sum())}", file=f)

    print(f"Saved caches and plots to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

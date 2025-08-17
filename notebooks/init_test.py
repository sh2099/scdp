# project_scdp_basis.py
import argparse
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Subset

# SCDP imports (assumes repository is on PYTHONPATH or installed)
from scdp.model.module import ChgLightningModule
from scdp.data.dataset import LmdbDataset  # same dataset used by test.py

def load_one_sample_from_dataset(data_path: Path, split_file: Path, idx: int = 0):
    # load dataset and splits similarly to test.py
    dataset = LmdbDataset(str(data_path))
    import json
    with open(split_file, 'r') as fp:
        splits = json.load(fp)
    test_subset = Subset(dataset, splits['test'])
    sample = test_subset[idx]  # dependent on LmdbDataset.__getitem__ API
    return sample

def build_basis_matrix_from_model(model: ChgLightningModule, sample):
    """
    Construct the design matrix B (n_probes x n_basis) for a single sample
    using the model.gto_dict produced by model.construct_orbitals().
    We evaluate each basis orbital individually by calling the corresponding
    GTOs module with a one-hot coeff for that orbital on that atom.
    """
    device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device('cpu')

    # Extract probe coords and true charge labels from the sample
    # sample may be a mapping or an object with attributes; handle both
    if hasattr(sample, 'probe_coords'):
        probe_coords = sample.probe_coords.detach().cpu().numpy()  # (M,3)
    else:
        probe_coords = sample['probe_coords'].detach().cpu().numpy()
    if hasattr(sample, 'chg_labels'):
        chg_labels = sample.chg_labels.detach().cpu().numpy()  # (M,)
    else:
        chg_labels = sample['chg_labels'].detach().cpu().numpy()

    # atom-level data
    if hasattr(sample, 'coords'):
        atom_coords = sample.coords.detach().cpu().numpy()  # (N,3)
    else:
        atom_coords = sample['coords'].detach().cpu().numpy()
    if hasattr(sample, 'atom_types'):
        atom_types = sample.atom_types.detach().cpu().numpy()  # (N,) atomic numbers
    else:
        atom_types = sample['atom_types'].detach().cpu().numpy()

    # prepare torch probe coords once
    coords_t = torch.tensor(probe_coords, dtype=torch.float32, device=device)  # (M,3)
    M = probe_coords.shape[0]

    # We'll collect columns in a list and then hstack
    columns = []
    col_meta = []  # store (atom_index, orbital_idx, element) for each column

    # iterate atoms (keep same order as sample.atom_types / coords)
    for atom_idx, (Z, atom_R) in enumerate(zip(atom_types, atom_coords)):
        gto_key = str(int(Z))
        if gto_key not in model.gto_dict:
            raise KeyError(f"Element {Z} not present in model.gto_dict")
        gto_module = model.gto_dict[gto_key]  # torch.nn.Module (GTOs instance)
        outdim = int(gto_module.outdim)  # number of orbitals per *atom* for this element

        # For each orbital of this atom, call GTOs with a one-hot coeff to get basis function at probe points
        for orb_j in range(outdim):
            # Create coeffs for the atoms-of-this-type input. We call GTOs with
            # `atom_coords` containing *just this one atom*, so n_atoms = 1
            # expected coeffs shape: (n_atoms, outdim) => (1, outdim)
            coeffs = torch.zeros((1, outdim), dtype=torch.float32, device=device)
            coeffs[0, orb_j] = 1.0

            # single atom coords as tensor
            atom_coords_t = torch.tensor(atom_R, dtype=torch.float32, device=device).unsqueeze(0)  # (1,3)

            # Call GTOs. We do not pass expo_scaling or pbc/cell for the simple non-periodic case.
            # The GTOs module will compute the contribution sum_j coeff_{atom,j} * phi_{atom,j}(r_probe).
            # With coeffs being a one-hot for a single atom we obtain phi_{atom,orb_j}(r_probe).
            # Output is expected to be shape (M,) (scalar charge contribution per probe).
            out = gto_module(
                probe_coords=coords_t,
                atom_coords=atom_coords_t,
                n_probes=torch.tensor([M], device=device),
                n_atoms=torch.tensor([1], device=device),
                coeffs=coeffs,
                expo_scaling=None,
                pbc=False,
                cell=None
            )

            # detach and convert to numpy column vector
            col = out.detach().cpu().numpy().ravel()  # (M,)
            columns.append(col)
            col_meta.append((atom_idx, orb_j, int(Z)))

    # Form design matrix B: shape (M, nbasis_total)
    B = np.vstack(columns).T  # (M, nbasis)
    return B, chg_labels, col_meta

def fit_coefficients(B, chg_labels, weights=None, rcond=None):
    """
    Weighted least squares using the probe points. If weights is None, treat as equal.
    B: (M, nbasis)
    chg_labels: (M,)
    """
    if weights is None:
        W_sqrt = np.ones((B.shape[0], 1))
    else:
        W_sqrt = np.sqrt(weights)[:, None]

    Aw = W_sqrt * B
    yw = (W_sqrt[:, 0] * chg_labels)
    c, *_ = np.linalg.lstsq(Aw, yw, rcond=rcond)
    return c

def main(ckpt_path, data_path, split_file, sample_index=0):
    # Load the model checkpoint (this will run construct_orbitals() and create model.gto_dict)
    print("Loading model from checkpoint:", ckpt_path)
    model = ChgLightningModule.load_from_checkpoint(str(ckpt_path))
    # put model on CPU (we only need GTO evaluation which uses tensors; adjust to cuda if desired)
    model = model.to('cpu')
    model.eval()

    # load dataset sample (use same splitting logic as test.py)
    sample = load_one_sample_from_dataset(data_path, split_file, idx=sample_index)

    # Build basis matrix using model.gto_dict
    print("Building basis matrix by evaluating SCDP GTOs at sample probe coords...")
    B, chg_labels, col_meta = build_basis_matrix_from_model(model, sample)
    print("Design matrix B shape:", B.shape)

    # For weights: SCDP uses probe_weights sometimes; try to read probe weights from sample if present
    weights = None
    if hasattr(sample, 'probe_weights'):
        weights = sample.probe_weights.detach().cpu().numpy()
    elif isinstance(sample, dict) and 'probe_weights' in sample:
        weights = sample['probe_weights'].detach().cpu().numpy()

    # Fit coefficients (weighted least squares)
    print("Fitting coefficients by weighted least-squares (lstsq)...")
    c = fit_coefficients(B, chg_labels, weights=weights)

    # Reconstruct and diagnose
    rho_rec = B.dot(c)
    mse = np.mean((rho_rec - chg_labels) ** 2)
    int_abs_err = np.sum(np.abs(rho_rec - chg_labels) * (weights if weights is not None else 1.0))
    ne_true = np.sum(chg_labels * (weights if weights is not None else 1.0))
    ne_rec = np.sum(rho_rec * (weights if weights is not None else 1.0))
    print(f"Fitting MSE on probe points: {mse:.6e}")
    print(f"Integrated absolute error (weighted): {int_abs_err:.6e}")
    print(f"Electron count true (weighted): {ne_true:.6e}")
    print(f"Electron count rec  (weighted): {ne_rec:.6e}")
    print("Number of basis functions fitted:", len(c))

    # Optionally: print first few coefficients with metadata
    print("First 10 fitted coefficients (atom_idx, orb_idx, Z):")
    for i in range(min(10, len(c))):
        ai, oj, Z = col_meta[i]
        print(f"col {i}: atom {ai}, orb {oj}, Z={Z}, c={c[i]:.6e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (folder or .ckpt file)")
    parser.add_argument("--data_path", required=True, help="Path to LMDB data dir")
    parser.add_argument("--split_file", required=True, help="JSON split file (like datasplits.json)")
    parser.add_argument("--sample_index", type=int, default=0, help="Index into the test subset to project")
    args = parser.parse_args()

    main(args.ckpt, args.data_path, args.split_file, args.sample_index)

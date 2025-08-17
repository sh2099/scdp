import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
import json
import csv
from pathlib import Path
import concurrent.futures
from threading import Lock

from notebooks.overlap_2 import (
    load_single_molecule, create_scdp_basis_functions, compute_overlap_integrals_gto, 
    analyze_overlaps_gto
)
from scdp.model.utils import get_nmape

# Set default dtype to double precision
torch.set_default_dtype(torch.float64)

class MultiGPUProcessor:
    """
    GPU processor that distributes work across multiple GPUs.
    """
    
    def __init__(self, device_list: List[str] = None):
        if device_list is None:
            # Auto-detect available GPUs
            if torch.cuda.is_available():
                self.devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
            else:
                self.devices = ['cpu']
        else:
            self.devices = device_list
        
        self.n_devices = len(self.devices)
        print(f"Initialized MultiGPUProcessor with {self.n_devices} devices: {self.devices}")
    
    def get_device(self, idx: int) -> str:
        """Get device by index."""
        return self.devices[idx % self.n_devices]
    
    def distribute_work(self, work_items):
        """Distribute work items across available devices."""
        work_per_device = [[] for _ in range(self.n_devices)]
        
        for i, item in enumerate(work_items):
            device_idx = i % self.n_devices
            work_per_device[device_idx].append(item)
        
        return work_per_device

def get_probe_chunks_multi_gpu(n_probes: int, max_n_probe_per_pass: int, n_gpus: int = 1):
    """
    Split probe points into chunks optimized for multi-GPU processing.
    
    Args:
        n_probes: Total number of probes
        max_n_probe_per_pass: Maximum probes per chunk
        n_gpus: Number of GPUs available
    
    Returns:
        List of probe index chunks
    """
    probe_indices = torch.arange(n_probes)
    
    # Calculate chunk size to utilize all GPUs efficiently
    chunk_size = min(max_n_probe_per_pass // n_gpus, n_probes // (n_gpus * 2))
    chunk_size = max(chunk_size, 1000)  # Minimum chunk size
    
    chunks = []
    start_idx = 0
    
    while start_idx < n_probes:
        end_idx = min(start_idx + chunk_size, n_probes)
        chunks.append(probe_indices[start_idx:end_idx])
        start_idx = end_idx
    
    return chunks

import concurrent.futures
from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from sklearn.metrics import r2_score, mean_absolute_error

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _ensure_device_index(device_str: str) -> int:
    """Return CUDA index from 'cuda:N' or raise for cpu."""
    dev = torch.device(device_str)
    if dev.type != "cuda":
        raise ValueError("Expected CUDA device, got: {}".format(device_str))
    if dev.index is None:
        # default device -> treat as 0
        return 0
    return dev.index

def _group_basis_by_element(basis_info: List[Dict]) -> Dict[int, List[Tuple[int, Dict]]]:
    basis_by_element: Dict[int, List[Tuple[int, Dict]]] = {}
    for i, bf in enumerate(basis_info):
        z = bf["atomic_number"]
        basis_by_element.setdefault(z, []).append((i, bf))
    return basis_by_element

def _pin_if_cuda(t: torch.Tensor, use_cuda: bool) -> torch.Tensor:
    return t.pin_memory() if use_cuda and t.is_cuda is False else t

# ------------------------------------------------------------
# Overlap integrals (vector O_mu)
# ------------------------------------------------------------

def compute_overlap_chunk_on_device(
    probe_coords_cpu: torch.Tensor,
    charge_density_cpu: torch.Tensor,
    probe_indices_cpu: torch.Tensor,
    basis_by_element: Dict[int, List[Tuple[int, Dict]]],
    gto_dict: Dict[int, any],
    volume_element,  # float or tensor
    device_str: str,
) -> torch.Tensor:
    """Compute overlap integral contribution for a probe chunk on a specific GPU."""
    dev = torch.device(device_str)
    use_cuda = (dev.type == "cuda")
    if use_cuda:
        torch.cuda.set_device(_ensure_device_index(device_str))

    # Move only the *block* to device (non_blocking if pinned)
    block_idx_cpu = probe_indices_cpu.long()
    probe_block = probe_coords_cpu.index_select(0, block_idx_cpu)
    rho_block = charge_density_cpu.index_select(0, block_idx_cpu)
    probe_block = probe_block.to(dev, non_blocking=use_cuda)
    rho_block = rho_block.to(dev, non_blocking=use_cuda)

    # Create per-device copy of GTOs (thread-safe)
    gto_dev = {z: deepcopy(gto_dict[z]).to(dev).double() for z in basis_by_element.keys()}

    n_block = probe_block.shape[0]
    overlaps_dev = torch.zeros(sum(len(v) for v in basis_by_element.values()), device=dev, dtype=torch.float64)

    # scalar volume element on device
    if isinstance(volume_element, torch.Tensor):
        vol = volume_element.to(dev, dtype=torch.float64)
    else:
        vol = torch.tensor(volume_element, device=dev, dtype=torch.float64)

    # Precreate small tensors once per block
    n_probes_tensor = torch.tensor([n_block], device=dev)
    n_atoms_tensor = torch.tensor([1], device=dev)

    # Fill overlaps by true global index "idx"
    for z, items in basis_by_element.items():
        gto = gto_dev[z]
        for idx, bf in items:
            atom_coord = bf["center"].unsqueeze(0).to(dev, dtype=torch.float64)
            orb_idx = bf["orbital_idx"]

            gto_vals = gto.forward(
                probe_coords=probe_block,
                atom_coords=atom_coord,
                n_probes=n_probes_tensor,
                n_atoms=n_atoms_tensor,
                coeffs=None,
                expo_scaling=None,
                reorder=False,
                pbc=False,
                cell=None,
            )
            # accumulate O_mu = sum_i rho(r_i) * omega_mu(r_i) * dV
            omega = gto_vals[:, orb_idx].to(torch.float64)
            overlaps_dev[idx] += torch.sum(rho_block * omega) * vol

    return overlaps_dev  # caller brings to CPU


def compute_overlap_integrals_gto_multi_gpu(
    molecule,
    gto_dict: Dict[int, any],
    basis_info: List[Dict],
    max_probes_per_block: int = 50_000,
    gpu_processor=None,
) -> Tuple[torch.Tensor, List[str]]:
    """Compute overlap integrals using multiple GPUs (threaded)."""
    if gpu_processor is None:
        gpu_processor = MultiGPUProcessor()

    n_basis = len(basis_info)
    print(f"Computing overlaps for {n_basis} GTO basis functions with {gpu_processor.n_devices} GPUs...")

    use_cuda = any("cuda" in d for d in gpu_processor.devices)

    # Keep master tensors on CPU, optionally pinned for faster H2D
    probe_coords_cpu = molecule.probe_coords.double()
    charge_density_cpu = molecule.chg_labels.double()
    probe_coords_cpu = _pin_if_cuda(probe_coords_cpu, use_cuda)
    charge_density_cpu = _pin_if_cuda(charge_density_cpu, use_cuda)
    n_probes = probe_coords_cpu.shape[0]

    # Volume element
    if hasattr(molecule, "grid_size"):
        grid_size = molecule.grid_size[0].double()
        cell_volume = torch.det(molecule.cell[0]).abs().double()
        volume_element = (cell_volume / torch.prod(grid_size)).item()
    else:
        cell_volume = torch.det(molecule.cell[0]).abs().double()
        volume_element = (cell_volume / n_probes).item()

    labels = [bf["label"] for bf in basis_info]
    basis_by_element = _group_basis_by_element(basis_info)

    # Build chunks and distribute
    probe_chunks = get_probe_chunks_multi_gpu(n_probes, max_probes_per_block, gpu_processor.n_devices)
    work_per_device = gpu_processor.distribute_work(probe_chunks)

    def _worker(device_idx, chunks):
        device_str = gpu_processor.get_device(device_idx)
        # Accumulate on device, then .cpu() at the end for merge
        accum = torch.zeros(n_basis, dtype=torch.float64, device=torch.device(device_str) if "cuda" in device_str else "cpu")
        if "cuda" in device_str:
            torch.cuda.set_device(_ensure_device_index(device_str))

        with torch.inference_mode():
            for k, idxs in enumerate(chunks):
                if len(idxs) == 0:
                    continue
                print(f"  GPU {device_idx}: overlap chunk {k+1}/{len(chunks)} ({len(idxs)} probes)")
                idxs_cpu = idxs if idxs.device.type == "cpu" else idxs.cpu()
                partial = compute_overlap_chunk_on_device(
                    probe_coords_cpu, charge_density_cpu, idxs_cpu,
                    basis_by_element, gto_dict, volume_element, device_str
                )
                # partial is on device_str
                accum += partial.to(accum.device, non_blocking=("cuda" in device_str))
        return accum.cpu()

    overlaps_cpu = torch.zeros(n_basis, dtype=torch.float64)
    with concurrent.futures.ThreadPoolExecutor(max_workers=gpu_processor.n_devices) as ex:
        futures = [ex.submit(_worker, i, chunks) for i, chunks in enumerate(work_per_device) if len(chunks) > 0]
        for fut in concurrent.futures.as_completed(futures):
            overlaps_cpu += fut.result()

    return overlaps_cpu, labels

# ------------------------------------------------------------
# Overlap matrix (W_{mu nu})
# ------------------------------------------------------------

def compute_overlap_matrix_chunk_on_device(
    probe_coords_cpu: torch.Tensor,
    probe_indices_cpu: torch.Tensor,
    basis_by_element: Dict[int, List[Tuple[int, Dict]]],
    gto_dict: Dict[int, any],
    volume_element,
    n_basis: int,
    device_str: str,
) -> torch.Tensor:
    """Compute overlap-matrix contribution for a chunk on a specific GPU."""
    dev = torch.device(device_str)
    use_cuda = (dev.type == "cuda")
    if use_cuda:
        torch.cuda.set_device(_ensure_device_index(device_str))

    block_idx_cpu = probe_indices_cpu.long()
    probe_block = probe_coords_cpu.index_select(0, block_idx_cpu)
    probe_block = probe_block.to(dev, non_blocking=use_cuda)

    # Thread-safe per-device GTO copies
    gto_dev = {z: deepcopy(gto_dict[z]).to(dev).double() for z in basis_by_element.keys()}

    # scalar dV on device
    vol = torch.tensor(volume_element, device=dev, dtype=torch.float64) if not isinstance(volume_element, torch.Tensor) else volume_element.to(dev, dtype=torch.float64)

    n_block = probe_block.shape[0]
    basis_values_block = torch.zeros(n_basis, n_block, device=dev, dtype=torch.float64)

    n_probes_tensor = torch.tensor([n_block], device=dev)
    n_atoms_tensor = torch.tensor([1], device=dev)

    # IMPORTANT: write rows by the *true* basis index "idx"
    with torch.inference_mode():
        for z, items in basis_by_element.items():
            gto = gto_dev[z]
            for idx, bf in items:
                atom_coord = bf["center"].unsqueeze(0).to(dev, dtype=torch.float64)
                orb_idx = bf["orbital_idx"]

                gto_vals = gto.forward(
                    probe_coords=probe_block,
                    atom_coords=atom_coord,
                    n_probes=n_probes_tensor,
                    n_atoms=n_atoms_tensor,
                    coeffs=None,
                    expo_scaling=None,
                    reorder=False,
                    pbc=False,
                    cell=None,
                )
                basis_values_block[idx] = gto_vals[:, orb_idx].to(torch.float64)

    chunk_matrix = torch.einsum("ik,jk->ij", basis_values_block, basis_values_block) * vol
    return chunk_matrix  # caller will .cpu() when merging


def compute_overlap_matrix_gto_multi_gpu(
    gto_dict: Dict[int, any],
    basis_info: List[Dict],
    molecule,
    max_probes_per_block: int = 50_000,
    gpu_processor=None,
) -> torch.Tensor:
    """Compute overlap matrix using multiple GPUs (threaded)."""
    if gpu_processor is None:
        gpu_processor = MultiGPUProcessor()

    n_basis = len(basis_info)
    print(f"Computing {n_basis}x{n_basis} overlap matrix with {gpu_processor.n_devices} GPUs...")

    use_cuda = any("cuda" in d for d in gpu_processor.devices)

    probe_coords_cpu = molecule.probe_coords.double()
    probe_coords_cpu = _pin_if_cuda(probe_coords_cpu, use_cuda)
    n_probes = probe_coords_cpu.shape[0]

    # Volume element
    if hasattr(molecule, "grid_size"):
        grid_size = molecule.grid_size[0].double()
        cell_volume = torch.det(molecule.cell[0]).abs().double()
        volume_element = (cell_volume / torch.prod(grid_size)).item()
    else:
        cell_volume = torch.det(molecule.cell[0]).abs().double()
        volume_element = (cell_volume / n_probes).item()

    basis_by_element = _group_basis_by_element(basis_info)

    probe_chunks = get_probe_chunks_multi_gpu(n_probes, max_probes_per_block, gpu_processor.n_devices)
    work_per_device = gpu_processor.distribute_work(probe_chunks)

    def _worker(device_idx, chunks):
        device_str = gpu_processor.get_device(device_idx)
        if "cuda" in device_str:
            torch.cuda.set_device(_ensure_device_index(device_str))
        local = torch.zeros(n_basis, n_basis, dtype=torch.float64, device=torch.device(device_str) if "cuda" in device_str else "cpu")

        with torch.inference_mode():
            for k, idxs in enumerate(chunks):
                if len(idxs) == 0:
                    continue
                print(f"  GPU {device_idx}: matrix chunk {k+1}/{len(chunks)} ({len(idxs)} probes)")
                idxs_cpu = idxs if idxs.device.type == "cpu" else idxs.cpu()
                partial = compute_overlap_matrix_chunk_on_device(
                    probe_coords_cpu, idxs_cpu, basis_by_element,
                    gto_dict, volume_element, n_basis, device_str
                )
                local += partial.to(local.device, non_blocking=("cuda" in device_str))
        return local.cpu()

    W_cpu = torch.zeros(n_basis, n_basis, dtype=torch.float64)
    with concurrent.futures.ThreadPoolExecutor(max_workers=gpu_processor.n_devices) as ex:
        futures = [ex.submit(_worker, i, chunks) for i, chunks in enumerate(work_per_device) if len(chunks) > 0]
        for fut in concurrent.futures.as_completed(futures):
            W_cpu += fut.result()

    return W_cpu

# ------------------------------------------------------------
# Reconstruction
# ------------------------------------------------------------

def reconstruct_charge_density_chunk_on_device(
    probe_coords_cpu: torch.Tensor,
    probe_indices_cpu: torch.Tensor,
    coefficients_dev: torch.Tensor,  # already on device
    basis_by_element: Dict[int, List[Tuple[int, Dict]]],
    gto_dict: Dict[int, any],
    device_str: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute density reconstruction for a chunk on a specific GPU."""
    dev = torch.device(device_str)
    use_cuda = (dev.type == "cuda")
    if use_cuda:
        torch.cuda.set_device(_ensure_device_index(device_str))

    idxs_cpu = probe_indices_cpu.long()
    probe_block = probe_coords_cpu.index_select(0, idxs_cpu).to(dev, non_blocking=use_cuda)
    density_block = torch.zeros(probe_block.shape[0], device=dev, dtype=torch.float64)

    gto_dev = {z: deepcopy(gto_dict[z]).to(dev).double() for z in basis_by_element.keys()}
    n_block = probe_block.shape[0]
    n_probes_tensor = torch.tensor([n_block], device=dev)
    n_atoms_tensor = torch.tensor([1], device=dev)

    with torch.inference_mode():
        for z, items in basis_by_element.items():
            gto = gto_dev[z]
            for idx, bf in items:
                atom_coord = bf["center"].unsqueeze(0).to(dev, dtype=torch.float64)
                orb_idx = bf["orbital_idx"]
                gto_vals = gto.forward(
                    probe_coords=probe_block,
                    atom_coords=atom_coord,
                    n_probes=n_probes_tensor,
                    n_atoms=n_atoms_tensor,
                    coeffs=None,
                    expo_scaling=None,
                    reorder=False,
                    pbc=False,
                    cell=None,
                )
                omega = gto_vals[:, orb_idx].to(torch.float64)
                # IMPORTANT: use the true global 'idx' for coefficients
                density_block += coefficients_dev[idx] * omega

    return density_block, idxs_cpu  # density on device; indices on CPU


def reconstruct_charge_density_multi_gpu(
    coefficients: torch.Tensor,
    gto_dict: Dict[int, any],
    basis_info: List[Dict],
    molecule,
    max_probes_per_block: int = 50_000,
    gpu_processor=None,
) -> torch.Tensor:
    """Reconstruct charge density using multiple GPUs (threaded)."""
    if gpu_processor is None:
        gpu_processor = MultiGPUProcessor()

    print(f"Reconstructing charge density with {gpu_processor.n_devices} GPUs...")

    use_cuda = any("cuda" in d for d in gpu_processor.devices)

    probe_coords_cpu = molecule.probe_coords.double()
    probe_coords_cpu = _pin_if_cuda(probe_coords_cpu, use_cuda)
    n_probes = probe_coords_cpu.shape[0]

    basis_by_element = _group_basis_by_element(basis_info)
    probe_chunks = get_probe_chunks_multi_gpu(n_probes, max_probes_per_block, gpu_processor.n_devices)
    work_per_device = gpu_processor.distribute_work(probe_chunks)

    recon_cpu = torch.zeros(n_probes, dtype=torch.float64)

    def _worker(device_idx, chunks):
        device_str = gpu_processor.get_device(device_idx)
        dev = torch.device(device_str)
        if "cuda" in device_str:
            torch.cuda.set_device(_ensure_device_index(device_str))
        # keep coefficients on the worker device
        coeff_dev = coefficients.to(dev, dtype=torch.float64, non_blocking=("cuda" in device_str))

        results = []
        with torch.inference_mode():
            for k, idxs in enumerate(chunks):
                if len(idxs) == 0:
                    continue
                print(f"  GPU {device_idx}: recon chunk {k+1}/{len(chunks)} ({len(idxs)} probes)")
                idxs_cpu = idxs if idxs.device.type == "cpu" else idxs.cpu()
                dens_dev, idxs_cpu_out = reconstruct_charge_density_chunk_on_device(
                    probe_coords_cpu, idxs_cpu, coeff_dev, basis_by_element, gto_dict, device_str
                )
                results.append((dens_dev.cpu(), idxs_cpu_out))
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=gpu_processor.n_devices) as ex:
        futures = [ex.submit(_worker, i, chunks) for i, chunks in enumerate(work_per_device) if len(chunks) > 0]
        for fut in concurrent.futures.as_completed(futures):
            for dens_cpu, idxs_cpu in fut.result():
                recon_cpu[idxs_cpu] = dens_cpu

    return recon_cpu

# ------------------------------------------------------------
# Analysis driver
# ------------------------------------------------------------

def analyze_single_molecule_multi_gpu(
    molecule_idx: int,
    regularizations: List[float],
    basis_set_name: str = 'def2-QZVPPD',
    use_augmentation: bool = True,
    beta: float = 2.0,
    max_probes_per_block: int = 50_000,
    gpu_processor=None,
    output_file: str = None,
) -> Dict:
    if gpu_processor is None:
        gpu_processor = MultiGPUProcessor()

    print(f"\n{'='*80}")
    print(f"ANALYZING MOLECULE {molecule_idx} WITH {gpu_processor.n_devices} GPUs")
    print(f"{'='*80}")

    # Load molecule & basis
    molecule = load_single_molecule(idx=molecule_idx)
    real_atoms_mask = molecule.atom_types != 0
    real_atom_types = molecule.atom_types[real_atoms_mask]
    real_atom_coords = molecule.coords[real_atoms_mask]

    gto_dict, basis_info = create_scdp_basis_functions(
        real_atom_types, real_atom_coords,
        basis_set_name=basis_set_name,
        use_augmentation=use_augmentation,
        beta=beta
    )
    print(f"Created {len(basis_info)} GTO basis functions")

    # Multi-GPU: overlaps and matrix
    print("Computing overlap integrals and matrix with multiple GPUs...")
    overlap_integrals, labels = compute_overlap_integrals_gto_multi_gpu(
        molecule, gto_dict, basis_info, max_probes_per_block, gpu_processor
    )
    overlap_matrix = compute_overlap_matrix_gto_multi_gpu(
        gto_dict, basis_info, molecule, max_probes_per_block, gpu_processor
    )

    results = {}
    for reg in regularizations:
        print(f"\n  Testing regularization {reg:.0e}...")
        try:
            primary_device = gpu_processor.get_device(0)
            dev = torch.device(primary_device) if 'cuda' in primary_device else torch.device('cpu')
            if dev.type == "cuda":
                torch.cuda.set_device(_ensure_device_index(primary_device))

            W = overlap_matrix.to(dev, dtype=torch.float64)
            b = overlap_integrals.to(dev, dtype=torch.float64)
            W_reg = W + torch.tensor(reg, dtype=torch.float64, device=dev) * torch.eye(W.shape[0], device=dev, dtype=torch.float64)

            with torch.inference_mode():
                coeffs = torch.linalg.solve(W_reg, b)

            # recon on multi-GPU; pass CPU copy
            recon = reconstruct_charge_density_multi_gpu(
                coeffs.cpu(), gto_dict, basis_info, molecule, max_probes_per_block, gpu_processor
            )

            # metrics (CPU)
            true_np = molecule.chg_labels.cpu().numpy()
            recon_np = recon.cpu().numpy()
            nmape = get_nmape(recon.double(), molecule.chg_labels.double()).item()
            r2 = r2_score(true_np, recon_np)
            mae = mean_absolute_error(true_np, recon_np)

            results[reg] = {
                'nmape': nmape,
                'r2': r2,
                'mae': mae,
                'coefficients': None,
                'reconstructed_density': None
            }
            print(f"    NMAPE: {nmape:.4f}, R²: {r2:.4f}")

        except Exception as e:
            print(f"    Failed with regularization {reg}: {e}")
            results[reg] = {'nmape': float('inf'), 'r2': -float('inf'), 'mae': float('inf'),
                            'coefficients': None, 'reconstructed_density': None}

    mol_result = {
        'molecule_idx': molecule_idx,
        'molecule': None,
        'results': results,
        'n_basis': len(basis_info),
    }

    if output_file:
        from notebooks.comp_chrg_multi import save_molecule_results
        save_molecule_results(mol_result, output_file)
        print(f"  Results saved to {output_file}")

    # Clean up
    del molecule, gto_dict, basis_info, overlap_integrals, overlap_matrix
    for dev_str in gpu_processor.devices:
        if 'cuda' in dev_str:
            with torch.cuda.device(torch.device(dev_str)):
                torch.cuda.empty_cache()

    return mol_result


def main():
    """Main function to run multi-GPU multi-molecule analysis."""
    print("="*80)
    print("MULTI-GPU MULTI-MOLECULE REGULARIZATION ANALYSIS")
    print("="*80)
    print("Using def2-QZVPPD basis set with multiple GPUs")
    print("Max probes per block: 50000")
    print("Using double precision (float64)")
    
    # Initialize multi-GPU processor
    gpu_processor = MultiGPUProcessor()
    
    # Configuration
    molecule_indices = np.random.choice(range(130000), size=50, replace=False).tolist()  # Example: 10
    regularizations = [1e-10]
    max_probes_per_block = 400000
    
    # Output file for results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"multi_gpu_molecule_results_{timestamp}.csv"
    
    print(f"Analyzing molecules: {molecule_indices}")
    print(f"Testing regularizations: {regularizations}")
    print(f"Using {gpu_processor.n_devices} GPUs: {gpu_processor.devices}")
    print(f"Results will be saved to: {results_file}")
    
    # Analyze each molecule
    total_start_time = time.time()
    
    for i, mol_idx in enumerate(molecule_indices):
        mol_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"PROCESSING MOLECULE {i+1}/{len(molecule_indices)} WITH MULTI-GPU")
        print(f"{'='*60}")
        
        try:
            mol_results = analyze_single_molecule_multi_gpu(
                mol_idx, regularizations,
                basis_set_name='def2-QZVPPD',
                use_augmentation=True,
                beta=2.0,
                max_probes_per_block=max_probes_per_block,
                gpu_processor=gpu_processor,
                output_file=results_file
            )
            
            mol_time = time.time() - mol_start_time
            print(f"Completed molecule {mol_idx} in {mol_time:.1f}s (multi-GPU)")
            
            # Free memory
            del mol_results
            
        except Exception as e:
            print(f"Failed to process molecule {mol_idx}: {e}")
            continue
    
    total_time = time.time() - total_start_time
    print(f"\nTotal multi-GPU analysis time: {total_time:.1f}s")
    
    # Create comprehensive analysis from saved file
    print(f"\n{'='*80}")
    print("CREATING ANALYSIS FROM SAVED RESULTS")
    print(f"{'='*80}")
    
    from notebooks.comp_chrg_multi import plot_multi_molecule_analysis_from_file
    summary = plot_multi_molecule_analysis_from_file(results_file)
    
    print(f"\nMulti-GPU analysis completed successfully!")
    print(f"Results saved to: {results_file}")
    print(f"Summary statistics available in 'summary' variable")
    print(f"Speedup achieved through {gpu_processor.n_devices} parallel GPUs")
    
    return results_file, summary

if __name__ == "__main__":
    results_file, summary = main()

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
    load_single_molecule, create_scdp_basis_functions
)
from scdp.model.utils import get_nmape

# Set default dtype to double precision
torch.set_default_dtype(torch.float64)

class MultiGPUProcessor:
    """
    GPU processor that distributes work across multiple GPUs.
    """
    
    def __init__(self, device_list: List[str] = None, exclude_gpus: List[int] = None):
        """
        Initialize MultiGPUProcessor with optional GPU exclusion.
        
        Args:
            device_list: List of device strings to use (if None, auto-detect)
            exclude_gpus: List of GPU indices to exclude from auto-detection
        """
        if device_list is None:
            # Auto-detect available GPUs
            if torch.cuda.is_available():
                all_gpus = list(range(torch.cuda.device_count()))
                if exclude_gpus is not None:
                    # Remove excluded GPUs
                    available_gpus = [i for i in all_gpus if i not in exclude_gpus]
                    if not available_gpus:
                        print("Warning: All GPUs excluded, falling back to CPU")
                        self.devices = ['cpu']
                    else:
                        self.devices = [f'cuda:{i}' for i in available_gpus]
                        if exclude_gpus:
                            print(f"Excluded GPUs: {exclude_gpus}")
                else:
                    self.devices = [f'cuda:{i}' for i in all_gpus]
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

def _group_basis_by_element_optimized(basis_info: List[Dict]) -> Dict[int, Dict]:
    """Group basis functions by element for optimized batched processing."""
    basis_by_element = {}
    for i, bf in enumerate(basis_info):
        z = bf["atomic_number"]
        if z not in basis_by_element:
            basis_by_element[z] = {
                'indices': [],
                'centers': [],
                'orbital_indices': [],
                'labels': []
            }
        basis_by_element[z]['indices'].append(i)
        basis_by_element[z]['centers'].append(bf["center"])
        basis_by_element[z]['orbital_indices'].append(bf["orbital_idx"])
        basis_by_element[z]['labels'].append(bf["label"])
    
    # Convert lists to tensors for efficient processing
    for z in basis_by_element:
        basis_by_element[z]['centers'] = torch.stack(basis_by_element[z]['centers'])
        basis_by_element[z]['orbital_indices'] = torch.tensor(basis_by_element[z]['orbital_indices'])
        basis_by_element[z]['indices'] = torch.tensor(basis_by_element[z]['indices'])
    
    return basis_by_element

def _pin_if_cuda(t: torch.Tensor, use_cuda: bool) -> torch.Tensor:
    return t.pin_memory() if use_cuda and t.is_cuda is False else t

# ------------------------------------------------------------
# Overlap integrals (vector O_mu) - OPTIMIZED
# ------------------------------------------------------------

def compute_overlap_chunk_on_device_optimized(
    probe_coords_cpu: torch.Tensor,
    charge_density_cpu: torch.Tensor,
    probe_indices_cpu: torch.Tensor,
    basis_by_element: Dict[int, Dict],
    gto_dict: Dict[int, any],
    volume_element,  # float or tensor
    device_str: str,
) -> torch.Tensor:
    """Compute overlap integral contribution for a probe chunk on a specific GPU - OPTIMIZED."""
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
    total_basis = sum(len(v['indices']) for v in basis_by_element.values())
    overlaps_dev = torch.zeros(total_basis, device=dev, dtype=torch.float64)

    # scalar volume element on device
    if isinstance(volume_element, torch.Tensor):
        vol = volume_element.to(dev, dtype=torch.float64)
    else:
        vol = torch.tensor(volume_element, device=dev, dtype=torch.float64)

    # Precreate tensors once per block
    n_probes_tensor = torch.tensor([n_block], device=dev)

    # OPTIMIZED: Process all basis functions of same element type together
    with torch.inference_mode():
        for z, z_data in basis_by_element.items():
            gto = gto_dev[z]
            n_atoms_z = len(z_data['centers'])
            if n_atoms_z == 0:
                continue
                
            n_atoms_tensor = torch.tensor([n_atoms_z], device=dev)
            atom_coords = z_data['centers'].to(dev, dtype=torch.float64)
            
            # Compute GTO values for all atoms of this type at once
            gto_vals = gto.forward(
                probe_coords=probe_block,
                atom_coords=atom_coords,
                n_probes=n_probes_tensor,
                n_atoms=n_atoms_tensor,
                coeffs=None,
                expo_scaling=None,
                reorder=False,
                pbc=False,
                cell=None,
            )
            # gto_vals shape: [n_probes, total_orbitals_for_this_element]
            
            # Extract values for each basis function using orbital indices
            orbital_indices = z_data['orbital_indices'].to(dev)
            global_indices = z_data['indices'].to(dev)
            
            for i, (orb_idx, global_idx) in enumerate(zip(orbital_indices, global_indices)):
                # Compute cumulative orbital offset for this atom
                atom_orbital_offset = i * gto.outdim
                omega = gto_vals[:, atom_orbital_offset + orb_idx].to(torch.float64)
                overlaps_dev[global_idx] += torch.sum(rho_block * omega) * vol

    return overlaps_dev  # caller brings to CPU

def compute_overlap_integrals_gto_multi_gpu(
    molecule,
    gto_dict: Dict[int, any],
    basis_info: List[Dict],
    max_probes_per_block: int = 50_000,
    gpu_processor=None,
) -> Tuple[torch.Tensor, List[str]]:
    """Compute overlap integrals using multiple GPUs (threaded) - OPTIMIZED."""
    if gpu_processor is None:
        gpu_processor = MultiGPUProcessor()

    n_basis = len(basis_info)
    print(f"Computing overlaps for {n_basis} GTO basis functions with {gpu_processor.n_devices} GPUs (OPTIMIZED)...")

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
    basis_by_element = _group_basis_by_element_optimized(basis_info)

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
                partial = compute_overlap_chunk_on_device_optimized(
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
# Overlap matrix (W_{mu nu}) - OPTIMIZED
# ------------------------------------------------------------

def compute_overlap_matrix_chunk_on_device_optimized(
    probe_coords_cpu: torch.Tensor,
    probe_indices_cpu: torch.Tensor,
    basis_by_element: Dict[int, Dict],
    gto_dict: Dict[int, any],
    volume_element,
    n_basis: int,
    device_str: str,
) -> torch.Tensor:
    """Compute overlap-matrix contribution for a chunk on a specific GPU - OPTIMIZED."""
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

    # Precreate tensors once per block
    n_probes_tensor = torch.tensor([n_block], device=dev)

    # OPTIMIZED: Process all basis functions of same element type together
    with torch.inference_mode():
        for z, z_data in basis_by_element.items():
            gto = gto_dev[z]
            n_atoms_z = len(z_data['centers'])
            if n_atoms_z == 0:
                continue
                
            n_atoms_tensor = torch.tensor([n_atoms_z], device=dev)
            atom_coords = z_data['centers'].to(dev, dtype=torch.float64)
            
            # Compute GTO values for all atoms of this type at once
            gto_vals = gto.forward(
                probe_coords=probe_block,
                atom_coords=atom_coords,
                n_probes=n_probes_tensor,
                n_atoms=n_atoms_tensor,
                coeffs=None,
                expo_scaling=None,
                reorder=False,
                pbc=False,
                cell=None,
            )
            # gto_vals shape: [n_probes, total_orbitals_for_this_element]
            
            # Extract values for each basis function using orbital indices
            orbital_indices = z_data['orbital_indices'].to(dev)
            global_indices = z_data['indices'].to(dev)
            
            for i, (orb_idx, global_idx) in enumerate(zip(orbital_indices, global_indices)):
                # Compute cumulative orbital offset for this atom
                atom_orbital_offset = i * gto.outdim
                basis_values_block[global_idx] = gto_vals[:, atom_orbital_offset + orb_idx].to(torch.float64)

    # Optimized symmetric matrix computation using outer product
    # W_ij = sum_k (phi_i(r_k) * phi_j(r_k) * dV)
    # This is equivalent to: basis_values @ basis_values.T * vol
    chunk_matrix = torch.matmul(basis_values_block, basis_values_block.T) * vol
    return chunk_matrix  # caller will .cpu() when merging

def compute_overlap_matrix_gto_multi_gpu(
    gto_dict: Dict[int, any],
    basis_info: List[Dict],
    molecule,
    max_probes_per_block: int = 50_000,
    gpu_processor=None,
) -> torch.Tensor:
    """Compute overlap matrix using multiple GPUs (threaded) with symmetry optimization - OPTIMIZED."""
    if gpu_processor is None:
        gpu_processor = MultiGPUProcessor()

    n_basis = len(basis_info)
    print(f"Computing {n_basis}x{n_basis} symmetric overlap matrix with {gpu_processor.n_devices} GPUs (OPTIMIZED)...")
    print("Optimizing for matrix symmetry and batched element processing...")

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

    basis_by_element = _group_basis_by_element_optimized(basis_info)

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
                partial = compute_overlap_matrix_chunk_on_device_optimized(
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

    # Ensure perfect symmetry by averaging W and W.T (handles numerical precision issues)
    print("Enforcing matrix symmetry...")
    W_cpu = (W_cpu + W_cpu.T) / 2.0
    
    # Verify symmetry
    symmetry_error = torch.max(torch.abs(W_cpu - W_cpu.T)).item()
    print(f"Matrix symmetry error: {symmetry_error:.2e}")
    
    return W_cpu

# ------------------------------------------------------------
# Reconstruction - OPTIMIZED
# ------------------------------------------------------------

def reconstruct_charge_density_chunk_on_device_optimized(
    probe_coords_cpu: torch.Tensor,
    probe_indices_cpu: torch.Tensor,
    coefficients_dev: torch.Tensor,  # already on device
    basis_by_element: Dict[int, Dict],
    gto_dict: Dict[int, any],
    device_str: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute density reconstruction for a chunk on a specific GPU - OPTIMIZED."""
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

    # OPTIMIZED: Process all basis functions of same element type together
    with torch.inference_mode():
        for z, z_data in basis_by_element.items():
            gto = gto_dev[z]
            n_atoms_z = len(z_data['centers'])
            if n_atoms_z == 0:
                continue
                
            n_atoms_tensor = torch.tensor([n_atoms_z], device=dev)
            atom_coords = z_data['centers'].to(dev, dtype=torch.float64)
            
            # Compute GTO values for all atoms of this type at once
            gto_vals = gto.forward(
                probe_coords=probe_block,
                atom_coords=atom_coords,
                n_probes=n_probes_tensor,
                n_atoms=n_atoms_tensor,
                coeffs=None,
                expo_scaling=None,
                reorder=False,
                pbc=False,
                cell=None,
            )
            # gto_vals shape: [n_probes, total_orbitals_for_this_element]
            
            # Extract values for each basis function using orbital indices
            orbital_indices = z_data['orbital_indices'].to(dev)
            global_indices = z_data['indices'].to(dev)
            
            for i, (orb_idx, global_idx) in enumerate(zip(orbital_indices, global_indices)):
                # Compute cumulative orbital offset for this atom
                atom_orbital_offset = i * gto.outdim
                omega = gto_vals[:, atom_orbital_offset + orb_idx].to(torch.float64)
                # IMPORTANT: use the true global 'idx' for coefficients
                density_block += coefficients_dev[global_idx] * omega

    return density_block, idxs_cpu  # density on device; indices on CPU

def reconstruct_charge_density_multi_gpu(
    coefficients: torch.Tensor,
    gto_dict: Dict[int, any],
    basis_info: List[Dict],
    molecule,
    max_probes_per_block: int = 50_000,
    gpu_processor=None,
) -> torch.Tensor:
    """Reconstruct charge density using multiple GPUs (threaded) - OPTIMIZED."""
    if gpu_processor is None:
        gpu_processor = MultiGPUProcessor()

    print(f"Reconstructing charge density with {gpu_processor.n_devices} GPUs (OPTIMIZED)...")

    use_cuda = any("cuda" in d for d in gpu_processor.devices)

    probe_coords_cpu = molecule.probe_coords.double()
    probe_coords_cpu = _pin_if_cuda(probe_coords_cpu, use_cuda)
    n_probes = probe_coords_cpu.shape[0]

    basis_by_element = _group_basis_by_element_optimized(basis_info)
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
                dens_dev, idxs_cpu_out = reconstruct_charge_density_chunk_on_device_optimized(
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
# Analysis driver - OPTIMIZED
# ------------------------------------------------------------

def analyze_single_molecule_multi_gpu(
    molecule_idx: int,
    regularizations: List[float],
    basis_set_name: str = 'def2-QZVPPD',
    use_augmentation: bool = True,
    beta: float = 2.0,
    max_probes_per_block: int = 50_000,
    use_vnodes: bool = False,
    override_atom_type: int = None,
    gpu_processor=None,
    exclude_gpus: List[int] = None,
    output_file: str = None,
) -> Dict:
    """
    Analyze a single molecule with multiple regularization values using multiple GPUs - OPTIMIZED.
    
    Args:
        molecule_idx: Index of molecule to analyze
        regularizations: List of regularization values to test
        basis_set_name: Basis set name
        use_augmentation: Whether to use augmentation
        beta: Beta parameter
        max_probes_per_block: Maximum probes per block
        use_vnodes: Whether to use virtual nodes
        override_atom_type: If specified, all atoms (real and virtual) will be set to this atomic number
        gpu_processor: Multi-GPU processor instance
        exclude_gpus: List of GPU indices to exclude (only used if gpu_processor is None)
        output_file: Path to save results (if None, results not saved)
    
    Returns:
        Dictionary with results for each regularization value
    """
    if gpu_processor is None:
        gpu_processor = MultiGPUProcessor(exclude_gpus=exclude_gpus)

    vnode_status = 'ON' if use_vnodes else 'OFF'
    override_status = f', override_Z={override_atom_type}' if override_atom_type is not None else ''
    print(f"\n{'='*80}")
    print(f"ANALYZING MOLECULE {molecule_idx} WITH {gpu_processor.n_devices} GPUs (OPTIMIZED, vnodes={vnode_status}{override_status})")
    print(f"{'='*80}")

    # Load molecule with or without virtual nodes
    molecule = load_single_molecule(idx=molecule_idx, vnode=use_vnodes)
    
    if use_vnodes:
        # Use both real atoms and virtual nodes for basis functions
        print("Using virtual nodes: placing basis functions on both atoms and vnodes")
        
        # Get real atoms (non-zero atom types)
        real_atoms_mask = molecule.atom_types != 0
        real_atom_types = molecule.atom_types[real_atoms_mask]
        real_atom_coords = molecule.coords[real_atoms_mask]
        
        # Get virtual nodes (if available)
        if hasattr(molecule, 'is_vnode'):
            vnode_mask = molecule.is_vnode
            vnode_coords = molecule.coords[vnode_mask]
            
            # Set virtual node atom types
            if override_atom_type is not None:
                # If overriding, use the specified type for vnodes
                vnode_types = override_atom_type * torch.ones(len(vnode_coords), dtype=torch.long)
            else:
                # Default: use hydrogen (Z=1) for virtual nodes
                vnode_types = 8 * torch.ones(len(vnode_coords), dtype=torch.long)
            
            # Combine real atoms and virtual nodes
            all_coords = torch.cat([real_atom_coords, vnode_coords], dim=0)
            all_types = torch.cat([real_atom_types, vnode_types], dim=0)
            
            print(f"Real atoms: {len(real_atom_types)}, Virtual nodes: {len(vnode_coords)}")
            print(f"Total basis centers: {len(all_coords)}")
        else:
            print("Warning: Virtual nodes requested but not found in molecule, using only real atoms")
            all_coords = real_atom_coords
            all_types = real_atom_types
    else:
        # Use only real atoms (original behavior)
        print("Using real atoms only")
        real_atoms_mask = molecule.atom_types != 0
        all_types = molecule.atom_types[real_atoms_mask]
        all_coords = molecule.coords[real_atoms_mask]
        print(f"Real atoms: {len(all_types)}")
    
    # Apply atom type override if specified
    if override_atom_type is not None:
        original_unique_types = all_types.unique().tolist()
        all_types = override_atom_type * torch.ones(len(all_coords), dtype=torch.long)
        print(f"ATOM TYPE OVERRIDE: Changed all {len(all_coords)} atoms from types {original_unique_types} to Z={override_atom_type}")
    else:
        print(f"Using original atom types: {all_types.unique().tolist()}")

    # Create GTO basis functions on all selected centers
    gto_dict, basis_info = create_scdp_basis_functions(
        all_types, all_coords,
        basis_set_name=basis_set_name,
        use_augmentation=use_augmentation,
        beta=beta
    )
    print(f"Created {len(basis_info)} GTO basis functions")

    # Multi-GPU: overlaps and matrix - OPTIMIZED
    print("Computing overlap integrals and matrix with multiple GPUs (OPTIMIZED)...")
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

            # recon on multi-GPU; pass CPU copy - OPTIMIZED
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
        'use_vnodes': use_vnodes,
        'override_atom_type': override_atom_type
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


def get_molecule_indices_from_csv(csv_file: str) -> List[int]:
    """
    Extract unique molecule indices from an existing CSV results file.
    
    Args:
        csv_file: Path to CSV file containing molecule results
        
    Returns:
        List of unique molecule indices found in the file
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_file)
        
        if 'molecule_idx' not in df.columns:
            raise ValueError(f"CSV file {csv_file} does not contain 'molecule_idx' column")
        
        # Get unique molecule indices and sort them
        molecule_indices = sorted(df['molecule_idx'].unique().tolist())
        
        print(f"Found {len(molecule_indices)} unique molecules in {csv_file}")
        print(f"Molecule index range: {min(molecule_indices)} to {max(molecule_indices)}")
        
        # Show first few indices as preview
        preview_count = min(10, len(molecule_indices))
        print(f"First {preview_count} molecules: {molecule_indices[:preview_count]}")
        if len(molecule_indices) > preview_count:
            print(f"... and {len(molecule_indices) - preview_count} more")
        
        return molecule_indices
        
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
        return []

def compare_csv_molecule_sets(csv_file1: str, csv_file2: str) -> Dict:
    """
    Compare molecule sets between two CSV files.
    
    Args:
        csv_file1: Path to first CSV file
        csv_file2: Path to second CSV file
        
    Returns:
        Dictionary with comparison results
    """
    molecules1 = set(get_molecule_indices_from_csv(csv_file1))
    molecules2 = set(get_molecule_indices_from_csv(csv_file2))
    
    common = molecules1.intersection(molecules2)
    only_in_1 = molecules1.difference(molecules2)
    only_in_2 = molecules2.difference(molecules1)
    
    comparison = {
        'file1': csv_file1,
        'file2': csv_file2,
        'molecules_file1': len(molecules1),
        'molecules_file2': len(molecules2),
        'common_molecules': len(common),
        'only_in_file1': len(only_in_1),
        'only_in_file2': len(only_in_2),
        'common_list': sorted(list(common)),
        'only_in_file1_list': sorted(list(only_in_1)),
        'only_in_file2_list': sorted(list(only_in_2))
    }
    
    print(f"\nCSV Molecule Set Comparison:")
    print(f"File 1 ({csv_file1}): {len(molecules1)} molecules")
    print(f"File 2 ({csv_file2}): {len(molecules2)} molecules")
    print(f"Common molecules: {len(common)}")
    print(f"Only in file 1: {len(only_in_1)}")
    print(f"Only in file 2: {len(only_in_2)}")
    
    if only_in_1:
        print(f"Missing from file 2: {sorted(list(only_in_1))[:10]}..." if len(only_in_1) > 10 else f"Missing from file 2: {sorted(list(only_in_1))}")
    if only_in_2:
        print(f"Missing from file 1: {sorted(list(only_in_2))[:10]}..." if len(only_in_2) > 10 else f"Missing from file 1: {sorted(list(only_in_2))}")
    
    return comparison

def main(use_vnodes: bool = False, reference_csv: str = None, override_atom_type: int = None, exclude_gpus: List[int] = None):
    """Main function to run multi-GPU multi-molecule analysis - OPTIMIZED."""
    print("="*80)
    print("MULTI-GPU MULTI-MOLECULE REGULARIZATION ANALYSIS (OPTIMIZED)")
    print("="*80)
    print("Using def2-QZVPPD basis set with multiple GPUs")
    print("Max probes per block: 50000")
    print("Using double precision (float64)")
    print("OPTIMIZATION: Batched processing per element type (eliminates slow basis function loops)")
    print(f"Virtual nodes: {'ENABLED' if use_vnodes else 'DISABLED'}")
    if override_atom_type is not None:
        print(f"Atom type override: All atoms will be set to Z={override_atom_type}")
    else:
        print("Atom type override: DISABLED (using original atom types)")
    if exclude_gpus:
        print(f"Excluded GPUs: {exclude_gpus}")
    
    # Check if we should run from existing CSV
    if reference_csv is not None:
        print(f"Running from reference CSV: {reference_csv}")
        molecule_indices = get_molecule_indices_from_csv(reference_csv)
        molecule_indices = molecule_indices[21:31]
    else:
        molecule_indices = [5, 343, 11797, 39941, 17]
    
    # Initialize multi-GPU processor with exclusions
    gpu_processor = MultiGPUProcessor(exclude_gpus=exclude_gpus)
    
    # Configuration
    regularizations = [1e-10]
    max_probes_per_block = 4000
    
    # Output file for results (include vnode and override status in filename)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    vnode_suffix = "_vnodes" if use_vnodes else ""
    override_suffix = f"_override{override_atom_type}" if override_atom_type is not None else ""
    results_file = f"multi_gpu_molecule_results_optimized{vnode_suffix}{override_suffix}_{timestamp}.csv"
    
    print(f"Analyzing molecules: {molecule_indices}")
    print(f"Testing regularizations: {regularizations}")
    print(f"Using {gpu_processor.n_devices} GPUs: {gpu_processor.devices}")
    print(f"Results will be saved to: {results_file}")
    
    # Analyze each molecule
    total_start_time = time.time()
    
    for i, mol_idx in enumerate(molecule_indices):
        mol_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"PROCESSING MOLECULE {i+1}/{len(molecule_indices)} WITH MULTI-GPU (OPTIMIZED)")
        print(f"{'='*60}")
        
        try:
            mol_results = analyze_single_molecule_multi_gpu(
                mol_idx, regularizations,
                basis_set_name='def2-QZVPPD',
                use_augmentation=True,
                beta=2.0,
                max_probes_per_block=max_probes_per_block,
                use_vnodes=use_vnodes,
                override_atom_type=override_atom_type,
                gpu_processor=gpu_processor,
                exclude_gpus=None,  # Already handled by gpu_processor
                output_file=results_file
            )
            
            mol_time = time.time() - mol_start_time
            print(f"Completed molecule {mol_idx} in {mol_time:.1f}s (multi-GPU optimized)")
            
            # Free memory
            del mol_results
            
        except Exception as e:
            print(f"Failed to process molecule {mol_idx}: {e}")
            continue
    
    total_time = time.time() - total_start_time
    print(f"\nTotal optimized multi-GPU analysis time: {total_time:.1f}s")
    
    # Create comprehensive analysis from saved file
    print(f"\n{'='*80}")
    print("CREATING ANALYSIS FROM SAVED RESULTS")
    print(f"{'='*80}")
    
    from notebooks.comp_chrg_multi import plot_multi_molecule_analysis_from_file
    summary = plot_multi_molecule_analysis_from_file(results_file)
    
    print(f"\nOptimized multi-GPU analysis completed successfully!")
    print(f"Results saved to: {results_file}")
    print(f"Summary statistics available in 'summary' variable")
    print(f"Speedup achieved through {gpu_processor.n_devices} parallel GPUs with optimized batching")
    print(f"Virtual nodes were {'ENABLED' if use_vnodes else 'DISABLED'}")
    if override_atom_type is not None:
        print(f"All atoms were overridden to Z={override_atom_type}")
    
    return results_file, summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-GPU multi-molecule charge density reconstruction analysis (OPTIMIZED)")
    parser.add_argument("--use_vnodes", action="store_true", 
                       help="Use virtual nodes for basis function placement")
    parser.add_argument("--reference_csv", type=str, default=None,
                       help="Path to existing CSV file to extract molecule indices from")
    parser.add_argument("--override_atom_type", type=int, default=None,
                       help="Override all atom types to this atomic number (e.g., 6 for carbon, 8 for oxygen)")
    parser.add_argument("--exclude_gpus", type=int, nargs='*', default=None,
                       help="GPU indices to exclude from use (e.g., --exclude_gpus 0 2)")
    args = parser.parse_args()
    
    results_file, summary = main(
        use_vnodes=args.use_vnodes, 
        reference_csv=args.reference_csv,
        override_atom_type=args.override_atom_type,
        exclude_gpus=args.exclude_gpus
    )
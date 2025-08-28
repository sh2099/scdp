import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from copy import deepcopy
import concurrent.futures

from scdp.model.basis_set import get_basis_set, transform_basis_set, aug_etb_for_basis
from scdp.model.gtos import GTOs
from scdp.model.utils import get_nmape

def get_probe_chunks(n_probes: int, max_n_probe_per_pass: int):
    """
    Split probe points into chunks for memory-efficient processing.
    Based on scdp's test.py implementation.
    
    Args:
        n_probes: Total number of probes
        max_n_probe_per_pass: Maximum probes per chunk
    
    Returns:
        List of probe index tensors for each chunk
    """
    probe_indices = torch.arange(n_probes)
    chunks = []
    
    start_idx = 0
    while start_idx < n_probes:
        end_idx = min(start_idx + max_n_probe_per_pass, n_probes)
        chunks.append(probe_indices[start_idx:end_idx])
        start_idx = end_idx
    
    return chunks

def get_molecule_probe_chunk(molecule, probe_indices):
    """
    Create a molecule chunk with subset of probes.
    Based on scdp's get_data_probe_chunk function.
    
    Args:
        molecule: Original molecule object
        probe_indices: Indices of probes to include
    
    Returns:
        Modified molecule object with subset of probes
    """
    mol_chunk = deepcopy(molecule)
    mol_chunk.chg_labels = molecule.chg_labels[probe_indices]
    mol_chunk.probe_coords = molecule.probe_coords[probe_indices]
    mol_chunk.n_probe = len(probe_indices)
    mol_chunk.sampled = True
    return mol_chunk

def create_gto_basis(atom_types: torch.Tensor, atom_coords: torch.Tensor,
                    basis_set_name: str = 'def2-QZVPPD', 
                    use_augmentation: bool = True,
                    beta: float = 2.0,
                    vnode_elem: int = 1,
                    silent: bool = False) -> Dict[str, GTOs]:
    """
    Create GTO basis using scdp's method, similar to ChgLightningModule.
    
    Args:
        atom_types: Atomic numbers for all centers
        atom_coords: Coordinates for all centers  
        basis_set_name: Basis set name
        use_augmentation: Whether to use even-tempered augmentation
        beta: Augmentation parameter
        vnode_elem: Element to use for virtual nodes (if atom_type is 0)
        silent: If True, suppress all print output
    
    Returns:
        Dictionary mapping atom type strings to GTO objects
    """
    if not silent:
        print(f"Creating GTO basis with {basis_set_name}")
        print(f"Augmentation: {use_augmentation}, beta: {beta}")
    
    # Get unique atom types
    unique_atom_types = torch.unique(atom_types).tolist()
    if not silent:
        print(f"Unique atom types: {unique_atom_types}")
    
    # Load and transform basis set
    basis_set = transform_basis_set(get_basis_set(basis_set_name))
    
    if use_augmentation:
        if not silent:
            print(f"Applying even-tempered augmentation with β={beta}")
        basis_set = aug_etb_for_basis(
            basis_set,
            beta=beta,
            lmax_restriction=True,
            lmax_relax=0
        )
    
    # Virtual node basis
    vbasis = basis_set[vnode_elem]
    
    # Create GTO dict following scdp convention
    gto_dict = {}
    for elem in unique_atom_types:
        if elem == 0:
            # Virtual nodes use specified element basis
            gto_dict['0'] = GTOs(**vbasis, cutoff=None, normalize=True)
            if not silent:
                print(f"Virtual nodes (type 0): using element {vnode_elem} basis")
        else:
            if elem in basis_set:
                gto_dict[str(elem)] = GTOs(**basis_set[elem], cutoff=None, normalize=True)
                if not silent:
                    print(f"Element {elem}: {gto_dict[str(elem)]}")
            else:
                if not silent:
                    print(f"Warning: Element {elem} not found in basis set")
    
    return gto_dict

def compute_overlap_integrals_scdp(molecule, gto_dict: Dict[str, GTOs], 
                                  atom_coords: torch.Tensor, atom_types: torch.Tensor,
                                  max_probes_per_chunk: int = 50000,
                                  devices: Optional[List[str]] = None,
                                  exclude_gpus: Optional[List[int]] = None,
                                  silent_gpu: bool = False) -> torch.Tensor:
    """
    Compute overlap integrals using scdp's GTO.compute() in a vectorized way
    with probe blocking. Now supports parallel processing across devices.
    
    Args:
        molecule: Molecule object with probe_coords and chg_labels
        gto_dict: Dictionary of GTO objects
        atom_coords: Coordinates of basis centers
        atom_types: Types of basis centers
        max_probes_per_chunk: Maximum probes per chunk
        devices: List of device strings to use (if None, auto-detect)
        exclude_gpus: List of GPU indices to exclude from auto-detection
        silent_gpu: If True, minimize GPU-related output
    
    Returns:
        Overlap integrals tensor
    """
    if not silent_gpu:
        print("Computing overlap integrals using scdp methods with probe blocking...")

    # Determine devices
    if devices is None:
        if torch.cuda.is_available():
            all_gpus = list(range(torch.cuda.device_count()))
            if exclude_gpus is not None:
                available_gpus = [i for i in all_gpus if i not in exclude_gpus]
                if not available_gpus:
                    if not silent_gpu:
                        print("Warning: All GPUs excluded, falling back to CPU")
                    devices = ["cpu"]
                else:
                    devices = [f"cuda:{i}" for i in available_gpus]
                    if exclude_gpus and not silent_gpu:
                        print(f"Excluded GPUs: {exclude_gpus}")
            else:
                devices = [f"cuda:{i}" for i in all_gpus]
        else:
            devices = ["cpu"]
    
    if not silent_gpu:
        print(f"Using devices: {devices}")

    # Get total number of probes
    n_probes = len(molecule.probe_coords)
    
    # Calculate volume element
    if hasattr(molecule, 'grid_size'):
        grid_size = molecule.grid_size[0].double()
        cell_volume = torch.det(molecule.cell[0]).abs().double()
        volume_element = cell_volume / torch.prod(grid_size)
    else:
        cell_volume = torch.det(molecule.cell[0]).abs().double()
        volume_element = cell_volume / n_probes
    
    atom_coords = atom_coords.double()
    unique_types = torch.unique(atom_types)

    # correct total basis functions (per-atom)
    total_basis_funcs = 0
    per_type_info = []
    for atom_type in unique_types:
        type_str = str(atom_type.item())
        if type_str in gto_dict:
            n_atoms_of_type = int((atom_types == atom_type).sum().item())
            outdim = gto_dict[type_str].outdim
            per_type_info.append((atom_type, type_str, n_atoms_of_type, outdim))
            total_basis_funcs += n_atoms_of_type * outdim

    overlaps_cpu = torch.zeros(total_basis_funcs, dtype=torch.float64)

    probe_chunks = get_probe_chunks(n_probes, max_probes_per_chunk)
    
    if silent_gpu:
        print(f"Processing {n_probes} probes in {len(probe_chunks)} chunks across {len(devices)} devices")
    else:
        print(f"Total probes: {n_probes}, chunks: {len(probe_chunks)}, total basis funcs: {total_basis_funcs}")

    # distribute chunks across devices (round-robin)
    n_devices = len(devices)
    work_per_device: List[List[torch.Tensor]] = [[] for _ in range(n_devices)]
    for i, chunk in enumerate(probe_chunks):
        work_per_device[i % n_devices].append(chunk)

    def _worker(device_str: str, chunks: List[torch.Tensor]) -> torch.Tensor:
        dev = torch.device(device_str)
        use_cuda = (dev.type == "cuda")
        if use_cuda:
            torch.cuda.set_device(dev)
        
        if not silent_gpu:
            print(f"[{device_str}] Worker starting with {len(chunks)} chunks")

        # create device-local copies
        gto_dev = {k: deepcopy(v).to(dev) for k, v in gto_dict.items()}
        atom_coords_dev = atom_coords.to(dev)

        local_accum = torch.zeros(total_basis_funcs, dtype=torch.float64, device=dev)

        for cidx, probe_indices in enumerate(chunks):
            if not silent_gpu:
                print(f"[{device_str}] processing chunk {cidx+1}/{len(chunks)} ({len(probe_indices)} probes)")
            mol_chunk = get_molecule_probe_chunk(molecule, probe_indices)
            probe_coords = mol_chunk.probe_coords.double().to(dev)
            charge_density = mol_chunk.chg_labels.double().to(dev)
            n_probes_chunk = len(probe_coords)

            basis_offset = 0
            for (atom_type, type_str, n_atoms_type, outdim) in per_type_info:
                if type_str not in gto_dev:
                    continue
                gto = gto_dev[type_str]
                type_mask = (atom_types == atom_type)
                type_coords = atom_coords_dev[type_mask]
                if len(type_coords) == 0:
                    continue

                # vectorized evaluation for all probe-atom pairs in this type
                vecs = (probe_coords.unsqueeze(1) - type_coords.unsqueeze(0)).reshape(-1, 3)
                vecs = vecs[..., [1,2,0]] / 0.52917721067
                orbital_pairs = gto.compute(vecs)  # (n_pairs, outdim)
                orbital_values = orbital_pairs.view(n_probes_chunk, len(type_coords), outdim)

                contrib = (charge_density.view(-1,1,1) * orbital_values).sum(dim=0) * volume_element

                # accumulate into local_accum
                for atom_idx in range(len(type_coords)):
                    start = basis_offset + atom_idx * outdim
                    end = start + outdim
                    local_accum[start:end] += contrib[atom_idx].to(local_accum.dtype)

                basis_offset += len(type_coords) * outdim

        if not silent_gpu:
            print(f"[{device_str}] Worker finished")
        return local_accum.cpu()

    # run workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_devices) as ex:
        futures = []
        for i, chunks in enumerate(work_per_device):
            if len(chunks) == 0:
                continue
            futures.append(ex.submit(_worker, devices[i], chunks))
        for fut in concurrent.futures.as_completed(futures):
            overlaps_cpu += fut.result()

    if not silent_gpu:
        print(f"Computed {len(overlaps_cpu)} overlap integrals")
        print(f"Sum of overlaps: {overlaps_cpu.sum():.6f}")
        print(f"Total electrons: {molecule.chg_labels.sum():.6f}")
    
    return overlaps_cpu

def compute_overlap_matrix_scdp(gto_dict: Dict[str, GTOs], 
                               atom_coords: torch.Tensor, atom_types: torch.Tensor,
                               molecule, max_probes_per_chunk: int = 50000,
                               devices: Optional[List[str]] = None,
                               exclude_gpus: Optional[List[int]] = None) -> torch.Tensor:
    """
    Compute overlap matrix using vectorized per-type calls to GTO.compute()
    and probe blocking. Parallel across devices.
    
    Args:
        gto_dict: Dictionary of GTO objects
        atom_coords: Coordinates of basis centers
        atom_types: Types of basis centers
        molecule: Molecule object for volume element calculation
        max_probes_per_chunk: Maximum probes per chunk
        devices: List of device strings to use (if None, auto-detect)
        exclude_gpus: List of GPU indices to exclude from auto-detection
    
    Returns:
        Overlap matrix tensor
    """
    print("Computing overlap matrix using scdp methods with probe blocking...")

    # determine devices
    if devices is None:
        if torch.cuda.is_available():
            all_gpus = list(range(torch.cuda.device_count()))
            if exclude_gpus is not None:
                available_gpus = [i for i in all_gpus if i not in exclude_gpus]
                if not available_gpus:
                    print("Warning: All GPUs excluded, falling back to CPU")
                    devices = ["cpu"]
                else:
                    devices = [f"cuda:{i}" for i in available_gpus]
                    if exclude_gpus:
                        print(f"Excluded GPUs: {exclude_gpus}")
            else:
                devices = [f"cuda:{i}" for i in all_gpus]
        else:
            devices = ["cpu"]
    print(f"Using devices: {devices}")

    n_probes = len(molecule.probe_coords)
    if hasattr(molecule, 'grid_size'):
        grid_size = molecule.grid_size[0].double()
        cell_volume = torch.det(molecule.cell[0]).abs().double()
        volume_element = cell_volume / torch.prod(grid_size)
    else:
        cell_volume = torch.det(molecule.cell[0]).abs().double()
        volume_element = cell_volume / n_probes

    atom_coords = atom_coords.double()
    unique_types = torch.unique(atom_types)

    # total basis funcs (per-atom)
    total_basis_funcs = 0
    per_type_info = []
    for atom_type in unique_types:
        type_str = str(atom_type.item())
        if type_str in gto_dict:
            n_atoms_of_type = int((atom_types == atom_type).sum().item())
            outdim = gto_dict[type_str].outdim
            per_type_info.append((atom_type, type_str, n_atoms_of_type, outdim))
            total_basis_funcs += n_atoms_of_type * outdim

    overlap_matrix_cpu = torch.zeros(total_basis_funcs, total_basis_funcs, dtype=torch.float64)

    probe_chunks = get_probe_chunks(n_probes, max_probes_per_chunk)
    n_devices = len(devices)
    work_per_device: List[List[torch.Tensor]] = [[] for _ in range(n_devices)]
    for i, chunk in enumerate(probe_chunks):
        work_per_device[i % n_devices].append(chunk)

    def _worker_mat(device_str: str, chunks: List[torch.Tensor]) -> torch.Tensor:
        dev = torch.device(device_str)
        use_cuda = (dev.type == "cuda")
        if use_cuda:
            torch.cuda.set_device(dev)
        #print(f"[{device_str}] Matrix worker starting with {len(chunks)} chunks")

        gto_dev = {k: deepcopy(v).to(dev) for k, v in gto_dict.items()}
        atom_coords_dev = atom_coords.to(dev)
        local_mat = torch.zeros(total_basis_funcs, total_basis_funcs, dtype=torch.float64, device=dev)

        for cidx, probe_indices in enumerate(chunks):
            print(f"[{device_str}] matrix chunk {cidx+1}/{len(chunks)} ({len(probe_indices)} probes)")
            mol_chunk = get_molecule_probe_chunk(molecule, probe_indices)
            probe_coords = mol_chunk.probe_coords.double().to(dev)
            n_probes_chunk = len(probe_coords)

            all_vals = torch.zeros(total_basis_funcs, n_probes_chunk, dtype=torch.float64, device=dev)
            basis_offset = 0
            for (atom_type, type_str, n_atoms_type, outdim) in per_type_info:
                if type_str not in gto_dev:
                    continue
                gto = gto_dev[type_str]
                type_mask = (atom_types == atom_type)
                type_coords = atom_coords_dev[type_mask]
                if len(type_coords) == 0:
                    continue

                vecs = (probe_coords.unsqueeze(1) - type_coords.unsqueeze(0)).reshape(-1, 3)
                vecs = vecs[..., [1,2,0]] / 0.52917721067
                orbital_pairs = gto.compute(vecs)
                orbital_values = orbital_pairs.view(n_probes_chunk, len(type_coords), outdim)

                for atom_idx in range(len(type_coords)):
                    start = basis_offset + atom_idx * outdim
                    end = start + outdim
                    all_vals[start:end] = orbital_values[:, atom_idx, :].T

                basis_offset += len(type_coords) * outdim

            chunk_overlap = torch.matmul(all_vals, all_vals.T) * volume_element
            local_mat += chunk_overlap

        print(f"[{device_str}] Matrix worker finished")
        return local_mat.cpu()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_devices) as ex:
        futures = []
        for i, chunks in enumerate(work_per_device):
            if len(chunks) == 0:
                continue
            futures.append(ex.submit(_worker_mat, devices[i], chunks))
        for fut in concurrent.futures.as_completed(futures):
            overlap_matrix_cpu += fut.result()

    overlap_matrix_cpu = (overlap_matrix_cpu + overlap_matrix_cpu.T) / 2.0
    print(f"Overlap matrix computed. Symmetry error: {torch.max(torch.abs(overlap_matrix_cpu - overlap_matrix_cpu.T)):.2e}")
    return overlap_matrix_cpu

def reconstruct_density_scdp(coefficients: torch.Tensor, gto_dict: Dict[str, GTOs],
                            atom_coords: torch.Tensor, atom_types: torch.Tensor,
                            molecule, max_probes_per_chunk: int = 50000,
                            devices: Optional[List[str]] = None,
                            exclude_gpus: Optional[List[int]] = None) -> torch.Tensor:
    """
    Reconstruct charge density using scdp methods with probe blocking and multi-GPU.
    
    Args:
        coefficients: Basis function coefficients
        gto_dict: Dictionary of GTO objects
        atom_coords: Coordinates of basis centers
        atom_types: Types of basis centers
        molecule: Molecule object
        max_probes_per_chunk: Maximum probes per chunk
        devices: List of device strings to use (if None, auto-detect)
        exclude_gpus: List of GPU indices to exclude from auto-detection
    
    Returns:
        Reconstructed charge density
    """
    print("Reconstructing charge density using scdp methods with probe blocking...")

    if devices is None:
        if torch.cuda.is_available():
            all_gpus = list(range(torch.cuda.device_count()))
            if exclude_gpus is not None:
                available_gpus = [i for i in all_gpus if i not in exclude_gpus]
                if not available_gpus:
                    print("Warning: All GPUs excluded, falling back to CPU")
                    devices = ["cpu"]
                else:
                    devices = [f"cuda:{i}" for i in available_gpus]
                    if exclude_gpus:
                        print(f"Excluded GPUs: {exclude_gpus}")
            else:
                devices = [f"cuda:{i}" for i in all_gpus]
        else:
            devices = ["cpu"]
    print(f"Using devices: {devices}")

    n_probes = len(molecule.probe_coords)
    reconstructed = torch.zeros(n_probes, dtype=torch.float64)

    atom_coords = atom_coords.double()
    unique_types = torch.unique(atom_types)

    probe_chunks = get_probe_chunks(n_probes, max_probes_per_chunk)
    n_devices = len(devices)
    work_per_device = [[] for _ in range(n_devices)]
    for i, chunk in enumerate(probe_chunks):
        work_per_device[i % n_devices].append(chunk)

    def _worker_recon(device_str: str, chunks: List[torch.Tensor]):
        dev = torch.device(device_str)
        use_cuda = (dev.type == "cuda")
        if use_cuda:
            torch.cuda.set_device(dev)
        #print(f"[{device_str}] Recon worker starting with {len(chunks)} chunks")

        gto_dev = {k: deepcopy(v).to(dev) for k, v in gto_dict.items()}
        atom_coords_dev = atom_coords.to(dev)
        coeff_dev = coefficients.to(dev)

        results = []  # list of tuples (probe_indices_cpu, dens_cpu)
        for cidx, probe_indices in enumerate(chunks):
            print(f"[{device_str}] recon chunk {cidx+1}/{len(chunks)} ({len(probe_indices)} probes)")
            mol_chunk = get_molecule_probe_chunk(molecule, probe_indices)
            probe_coords = mol_chunk.probe_coords.double().to(dev)
            n_probes_chunk = len(probe_coords)
            chunk_reconstructed = torch.zeros(n_probes_chunk, dtype=torch.float64, device=dev)

            basis_offset = 0
            for atom_type in unique_types:
                type_str = str(atom_type.item())
                if type_str not in gto_dev:
                    continue
                gto = gto_dev[type_str]
                type_mask = (atom_types == atom_type)
                type_coords = atom_coords_dev[type_mask]
                n_atoms_type = len(type_coords)
                if n_atoms_type == 0:
                    continue

                type_coeffs = coeff_dev[basis_offset:basis_offset + n_atoms_type * gto.outdim]
                type_coeffs = type_coeffs.view(n_atoms_type, gto.outdim)

                n_probes_tensor = torch.tensor([n_probes_chunk], device=dev)
                n_atoms_tensor = torch.tensor([n_atoms_type], device=dev)

                type_contribution = gto.forward(
                    probe_coords=probe_coords,
                    atom_coords=type_coords,
                    n_probes=n_probes_tensor,
                    n_atoms=n_atoms_tensor,
                    coeffs=type_coeffs,
                    expo_scaling=None,
                    reorder=True,
                    pbc=False,
                    cell=None
                )
                chunk_reconstructed += type_contribution
                basis_offset += n_atoms_type * gto.outdim

            results.append((probe_indices.cpu(), chunk_reconstructed.cpu()))

        print(f"[{device_str}] Recon worker finished")
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_devices) as ex:
        futures = []
        for i, chunks in enumerate(work_per_device):
            if len(chunks) == 0:
                continue
            futures.append(ex.submit(_worker_recon, devices[i], chunks))

        for fut in concurrent.futures.as_completed(futures):
            for probe_idx_cpu, dens_cpu in fut.result():
                reconstructed[probe_idx_cpu] = dens_cpu

    return reconstructed

def compute_overlap_integrals_2d_scdp(molecule, gto_dict: Dict[str, GTOs], 
                                     atom_coords: torch.Tensor, atom_types: torch.Tensor,
                                     max_probes_per_chunk: int = 50000,
                                     devices: Optional[List[str]] = None,
                                     exclude_gpus: Optional[List[int]] = None,
                                     silent_gpu: bool = False,
                                     silent_mapping: bool = False) -> Tuple[torch.Tensor, Dict]:
    """
    Compute overlap integrals as 2D matrix: exponents × (atoms, L, m hierarchy).
    
    Args:
        silent_gpu: If True, minimize GPU-related output
        silent_mapping: If True, minimize basis mapping output
    
    Returns:
        overlap_matrix_2d: Shape (n_exponents, n_basis_functions)
        mapping_info: Dictionary with mapping and structural information
    """
    if not silent_mapping:
        print("Computing 2D overlap integrals organized by exponents...")
    
    # Get basis function mapping
    mapping, exponent_to_basis, basis_info, atom_basis_structure = get_basis_function_mapping(
        gto_dict, atom_types, silent=silent_mapping
    )
    n_exponents = basis_info['n_exponents']
    n_basis_funcs = basis_info['total_basis_functions']
    
    if not silent_mapping:
        print(f"Basis structure: {n_exponents} exponents × {n_basis_funcs} basis functions")
    
    # Compute regular overlap integrals
    overlap_1d = compute_overlap_integrals_scdp(
        molecule, gto_dict, atom_coords, atom_types, 
        max_probes_per_chunk, devices, exclude_gpus, silent_gpu
    )
    
    # Reshape into 2D matrix
    overlap_2d = torch.zeros(n_exponents, n_basis_funcs, dtype=torch.float64)
    
    for exp_idx in range(n_exponents):
        if exp_idx in exponent_to_basis:
            basis_indices = exponent_to_basis[exp_idx]
            overlap_2d[exp_idx, basis_indices] = overlap_1d[basis_indices]
    
    mapping_info = {
        'mapping': mapping,
        'exponent_to_basis': exponent_to_basis,
        'basis_info': basis_info,
        'exponent_values': basis_info['exponent_values'],
        'atom_basis_structure': atom_basis_structure
    }
    
    return overlap_2d, mapping_info

def get_basis_function_mapping(gto_dict: Dict[str, GTOs], atom_types: torch.Tensor, silent: bool = False):
    """
    Create mapping from flat basis function index to (atom_idx, L, m, exponent_idx).
    
    Args:
        silent: If True, minimize output
    """
    mapping = []
    exponent_to_basis = {}
    atom_basis_structure = {}
    
    basis_idx = 0
    unique_types = torch.unique(atom_types)
    
    # First pass: collect all unique exponents across all atom types
    all_exponents = set()
    reference_gto = None
    
    for atom_type in unique_types:
        type_str = str(atom_type.item())
        if type_str in gto_dict:
            gto = gto_dict[type_str]
            expos = gto.expos.cpu().numpy()
            Ls = gto.Ls.cpu().numpy()
            
            if reference_gto is None:
                reference_gto = gto
                all_exponents.update(expos)
                if not silent:
                    print(f"Basis mapping reference (atom type {atom_type.item()}):")
                    print(f"  {len(expos)} exponents: {expos.min():.3e} to {expos.max():.3e}")
                    print(f"  L values: {sorted(set(Ls))}")
            else:
                # Verify this GTO has the same exponents as the reference
                ref_expos = reference_gto.expos.cpu().numpy()
                ref_Ls = reference_gto.Ls.cpu().numpy()
                
                if not (np.allclose(expos, ref_expos, rtol=1e-10) and np.array_equal(Ls, ref_Ls)):
                    if not silent:
                        print(f"WARNING: Atom type {atom_type.item()} has different basis!")
                else:
                    if not silent:
                        print(f"✓ Atom type {atom_type.item()} matches reference basis")
                    
                all_exponents.update(expos)
    
    # Sort exponents for consistent ordering
    sorted_exponents = sorted(all_exponents)
    exp_to_idx = {exp: idx for idx, exp in enumerate(sorted_exponents)}
    
    if not silent:
        print(f"\nBasis verification summary:")
        print(f"  Total unique atom types: {len(unique_types)}")
        print(f"  Final exponent set: {len(sorted_exponents)} unique exponents")
        print(f"  Range: {min(sorted_exponents):.3e} to {max(sorted_exponents):.3e}")
    
    # Second pass: build mapping and structure
    for atom_idx, atom_type in enumerate(atom_types):
        type_str = str(atom_type.item())
        if type_str not in gto_dict:
            continue
            
        atom_basis_structure[atom_idx] = {}
        gto = gto_dict[type_str]
        Ls = gto.Ls.cpu().numpy()
        expos = gto.expos.cpu().numpy()
        
        # Group by contraction (if available)
        if hasattr(gto, 'contraction') and gto.contraction is not None:
            contractions = gto.contraction.cpu().numpy()
            unique_contractions = sorted(set(contractions))
            
            for contraction_idx in unique_contractions:
                mask = contractions == contraction_idx
                L = Ls[mask][0]  # All primitives in same contraction have same L
                
                if L not in atom_basis_structure[atom_idx]:
                    atom_basis_structure[atom_idx][L] = {}
                
                # All m values for this L
                for m in range(-L, L + 1):
                    if m not in atom_basis_structure[atom_idx][L]:
                        atom_basis_structure[atom_idx][L][m] = []
                    
                    for prim_idx in range(len(Ls[mask])):
                        exp_val = expos[mask][prim_idx]
                        exp_idx = exp_to_idx[exp_val]
                        
                        mapping.append((atom_idx, L, m, exp_idx, exp_val))
                        atom_basis_structure[atom_idx][L][m].append({
                            'basis_idx': basis_idx,
                            'exp_idx': exp_idx,
                            'exp_val': exp_val
                        })
                        
                        if exp_idx not in exponent_to_basis:
                            exponent_to_basis[exp_idx] = []
                        exponent_to_basis[exp_idx].append(basis_idx)
                        
                        basis_idx += 1
        else:
            # No contraction - each primitive is separate
            for i, (L, exp_val) in enumerate(zip(Ls, expos)):
                exp_idx = exp_to_idx[exp_val]
                
                if L not in atom_basis_structure[atom_idx]:
                    atom_basis_structure[atom_idx][L] = {}
                
                for m in range(-L, L + 1):
                    if m not in atom_basis_structure[atom_idx][L]:
                        atom_basis_structure[atom_idx][L][m] = []
                    
                    mapping.append((atom_idx, L, m, exp_idx, exp_val))
                    atom_basis_structure[atom_idx][L][m].append({
                        'basis_idx': basis_idx,
                        'exp_idx': exp_idx,
                        'exp_val': exp_val
                    })
                    
                    if exp_idx not in exponent_to_basis:
                        exponent_to_basis[exp_idx] = []
                    exponent_to_basis[exp_idx].append(basis_idx)
                    
                    basis_idx += 1
    
    basis_info = {
        'total_basis_functions': basis_idx,
        'n_exponents': len(sorted_exponents),
        'exponent_values': sorted_exponents,
        'n_atoms': len(atom_types)
    }
    
    if not silent:
        print(f"Final verification: {len(sorted_exponents)} exponents mapped to {basis_idx} basis functions")
    
    return mapping, exponent_to_basis, basis_info, atom_basis_structure

def compute_overlap_integrals_2d_scdp(molecule, gto_dict: Dict[str, GTOs], 
                                     atom_coords: torch.Tensor, atom_types: torch.Tensor,
                                     max_probes_per_chunk: int = 50000,
                                     devices: Optional[List[str]] = None,
                                     exclude_gpus: Optional[List[int]] = None,
                                     silent_gpu: bool = False,
                                     silent_mapping: bool = False) -> Tuple[torch.Tensor, Dict]:
    """
    Compute overlap integrals as 2D matrix: exponents × (atoms, L, m hierarchy).
    
    Args:
        silent_gpu: If True, minimize GPU-related output
        silent_mapping: If True, minimize basis mapping output
    
    Returns:
        overlap_matrix_2d: Shape (n_exponents, n_basis_functions)
        mapping_info: Dictionary with mapping and structural information
    """
    if not silent_mapping:
        print("Computing 2D overlap integrals organized by exponents...")
    
    # Get basis function mapping
    mapping, exponent_to_basis, basis_info, atom_basis_structure = get_basis_function_mapping(
        gto_dict, atom_types, silent=silent_mapping
    )
    n_exponents = basis_info['n_exponents']
    n_basis_funcs = basis_info['total_basis_functions']
    
    if not silent_mapping:
        print(f"Basis structure: {n_exponents} exponents × {n_basis_funcs} basis functions")
    
    # Compute regular overlap integrals
    overlap_1d = compute_overlap_integrals_scdp(
        molecule, gto_dict, atom_coords, atom_types, 
        max_probes_per_chunk, devices, exclude_gpus, silent_gpu
    )
    
    # Reshape into 2D matrix
    overlap_2d = torch.zeros(n_exponents, n_basis_funcs, dtype=torch.float64)
    
    for exp_idx in range(n_exponents):
        if exp_idx in exponent_to_basis:
            basis_indices = exponent_to_basis[exp_idx]
            overlap_2d[exp_idx, basis_indices] = overlap_1d[basis_indices]
    
    mapping_info = {
        'mapping': mapping,
        'exponent_to_basis': exponent_to_basis,
        'basis_info': basis_info,
        'exponent_values': basis_info['exponent_values'],
        'atom_basis_structure': atom_basis_structure
    }
    
    return overlap_2d, mapping_info


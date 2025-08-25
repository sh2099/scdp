import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

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
                    vnode_elem: int = 1) -> Dict[str, GTOs]:
    """
    Create GTO basis using scdp's method, similar to ChgLightningModule.
    
    Args:
        atom_types: Atomic numbers for all centers
        atom_coords: Coordinates for all centers  
        basis_set_name: Basis set name
        use_augmentation: Whether to use even-tempered augmentation
        beta: Augmentation parameter
        vnode_elem: Element to use for virtual nodes (if atom_type is 0)
    
    Returns:
        Dictionary mapping atom type strings to GTO objects
    """
    print(f"Creating GTO basis with {basis_set_name}")
    print(f"Augmentation: {use_augmentation}, beta: {beta}")
    
    # Get unique atom types
    unique_atom_types = torch.unique(atom_types).tolist()
    print(f"Unique atom types: {unique_atom_types}")
    
    # Load and transform basis set
    basis_set = transform_basis_set(get_basis_set(basis_set_name))
    
    if use_augmentation:
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
            print(f"Virtual nodes (type 0): using element {vnode_elem} basis")
        else:
            if elem in basis_set:
                gto_dict[str(elem)] = GTOs(**basis_set[elem], cutoff=None, normalize=True)
                print(f"Element {elem}: {gto_dict[str(elem)]}")
            else:
                print(f"Warning: Element {elem} not found in basis set")
    
    return gto_dict

def compute_overlap_integrals_scdp(molecule, gto_dict: Dict[str, GTOs], 
                                  atom_coords: torch.Tensor, atom_types: torch.Tensor,
                                  max_probes_per_chunk: int = 50000) -> torch.Tensor:
    """
    Compute overlap integrals using scdp's GTO.compute() in a vectorized way
    with probe blocking. Uses per-atom ordering: for each atom of a given
    element we append its outdim orbitals in sequence.
    """
    print("Computing overlap integrals using scdp methods with probe blocking...")
    
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
    
    print(f"Volume element: {volume_element:.6f}")
    print(f"Total probes: {n_probes}")
    print(f"Max probes per chunk: {max_probes_per_chunk}")
    
    atom_coords = atom_coords.double()
    unique_types = torch.unique(atom_types)
    
    # Correct total basis functions: sum over atoms (not unique types)
    total_basis_funcs = 0
    for atom_type in unique_types:
        type_str = str(atom_type.item())
        if type_str in gto_dict:
            n_atoms_of_type = int((atom_types == atom_type).sum().item())
            total_basis_funcs += n_atoms_of_type * gto_dict[type_str].outdim

    overlaps = torch.zeros(total_basis_funcs, dtype=torch.float64)
    print(f"Total basis functions: {total_basis_funcs}")
    
    probe_chunks = get_probe_chunks(n_probes, max_probes_per_chunk)
    print(f"Processing {len(probe_chunks)} probe chunks...")
    
    for chunk_idx, probe_indices in enumerate(probe_chunks):
        print(f"  Processing chunk {chunk_idx + 1}/{len(probe_chunks)}: {len(probe_indices)} probes")
        mol_chunk = get_molecule_probe_chunk(molecule, probe_indices)
        probe_coords = mol_chunk.probe_coords.double()          # (n_probes_chunk, 3)
        charge_density = mol_chunk.chg_labels.double()         # (n_probes_chunk,)
        n_probes_chunk = len(probe_coords)
        
        basis_offset = 0
        for atom_type in unique_types:
            type_str = str(atom_type.item())
            if type_str not in gto_dict:
                continue
            gto = gto_dict[type_str]
            type_mask = (atom_types == atom_type)
            type_coords = atom_coords[type_mask]                # (n_atoms_type, 3)
            n_atoms_type = len(type_coords)
            if n_atoms_type == 0:
                continue
            
            # Vectorized per-type evaluation:
            # build vecs: (n_probes_chunk, n_atoms_type, 3) -> flatten to (n_pairs,3)
            vecs = (probe_coords.unsqueeze(1) - type_coords.unsqueeze(0)).reshape(-1, 3)
            # reorder and convert to bohr
            vecs = vecs[..., [1,2,0]] / 0.52917721067
            # compute orbital values for all pairs: (n_pairs, outdim)
            orbital_pairs = gto.compute(vecs)  # (n_probes_chunk * n_atoms_type, outdim)
            # reshape to (n_probes_chunk, n_atoms_type, outdim)
            orbital_values = orbital_pairs.view(n_probes_chunk, n_atoms_type, gto.outdim)
            
            # integrate: sum over probes of rho * phi  -> (n_atoms_type, outdim)
            contrib = (charge_density.view(-1,1,1) * orbital_values).sum(dim=0) * volume_element
            
            # place per-atom contributions into overlaps vector
            for atom_idx in range(n_atoms_type):
                start = basis_offset + atom_idx * gto.outdim
                end = start + gto.outdim
                overlaps[start:end] += contrib[atom_idx]
            
            basis_offset += n_atoms_type * gto.outdim
    
    print(f"Computed {len(overlaps)} overlap integrals")
    print(f"Sum of overlaps: {overlaps.sum():.6f}")
    print(f"Total electrons: {molecule.chg_labels.sum():.6f}")
    return overlaps

def compute_overlap_matrix_scdp(gto_dict: Dict[str, GTOs], 
                               atom_coords: torch.Tensor, atom_types: torch.Tensor,
                               molecule, max_probes_per_chunk: int = 50000) -> torch.Tensor:
    """
    Compute overlap matrix using vectorized per-type calls to GTO.compute()
    and probe blocking. Basis ordering matches compute_overlap_integrals_scdp.
    """
    print("Computing overlap matrix using scdp methods with probe blocking...")
    
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
    
    # Correct total basis functions
    total_basis_funcs = 0
    for atom_type in unique_types:
        type_str = str(atom_type.item())
        if type_str in gto_dict:
            n_atoms_of_type = int((atom_types == atom_type).sum().item())
            total_basis_funcs += n_atoms_of_type * gto_dict[type_str].outdim

    overlap_matrix = torch.zeros(total_basis_funcs, total_basis_funcs, dtype=torch.float64)
    print(f"Computing {total_basis_funcs}x{total_basis_funcs} overlap matrix")
    print(f"Total probes: {n_probes}")
    print(f"Max probes per chunk: {max_probes_per_chunk}")
    
    probe_chunks = get_probe_chunks(n_probes, max_probes_per_chunk)
    print(f"Processing {len(probe_chunks)} probe chunks...")
    
    for chunk_idx, probe_indices in enumerate(probe_chunks):
        print(f"  Processing chunk {chunk_idx + 1}/{len(probe_chunks)}: {len(probe_indices)} probes")
        mol_chunk = get_molecule_probe_chunk(molecule, probe_indices)
        probe_coords = mol_chunk.probe_coords.double()
        n_probes_chunk = len(probe_coords)
        
        all_basis_values = torch.zeros(total_basis_funcs, n_probes_chunk, dtype=torch.float64)
        basis_offset = 0
        
        for atom_type in unique_types:
            type_str = str(atom_type.item())
            if type_str not in gto_dict:
                continue
            gto = gto_dict[type_str]
            type_mask = (atom_types == atom_type)
            type_coords = atom_coords[type_mask]
            n_atoms_type = len(type_coords)
            if n_atoms_type == 0:
                continue
            
            # Vectorized per-type evaluation
            vecs = (probe_coords.unsqueeze(1) - type_coords.unsqueeze(0)).reshape(-1, 3)
            vecs = vecs[..., [1,2,0]] / 0.52917721067
            orbital_pairs = gto.compute(vecs)  # (n_pairs, outdim)
            orbital_values = orbital_pairs.view(n_probes_chunk, n_atoms_type, gto.outdim)
            
            for atom_idx in range(n_atoms_type):
                atom_basis_start = basis_offset + atom_idx * gto.outdim
                atom_basis_end = atom_basis_start + gto.outdim
                if atom_basis_end > total_basis_funcs:
                    raise IndexError(f"Basis index {atom_basis_end-1} out of bounds for size {total_basis_funcs}")
                all_basis_values[atom_basis_start:atom_basis_end] = orbital_values[:, atom_idx, :].T
            
            basis_offset += n_atoms_type * gto.outdim
        
        chunk_overlap = torch.matmul(all_basis_values, all_basis_values.T) * volume_element
        overlap_matrix += chunk_overlap
    
    overlap_matrix = (overlap_matrix + overlap_matrix.T) / 2.0
    print(f"Overlap matrix computed. Symmetry error: {torch.max(torch.abs(overlap_matrix - overlap_matrix.T)):.2e}")
    return overlap_matrix

def reconstruct_density_scdp(coefficients: torch.Tensor, gto_dict: Dict[str, GTOs],
                            atom_coords: torch.Tensor, atom_types: torch.Tensor,
                            molecule, max_probes_per_chunk: int = 50000) -> torch.Tensor:
    """
    Reconstruction remains efficient by using GTO.forward with coeffs per-type.
    Ordering of coefficients must match per-atom ordering used above.
    """
    print("Reconstructing charge density using scdp methods with probe blocking...")
    
    n_probes = len(molecule.probe_coords)
    reconstructed = torch.zeros(n_probes, dtype=torch.float64)
    
    atom_coords = atom_coords.double()
    unique_types = torch.unique(atom_types)
    
    print(f"Total probes: {n_probes}")
    print(f"Max probes per chunk: {max_probes_per_chunk}")
    
    # Get probe chunks
    probe_chunks = get_probe_chunks(n_probes, max_probes_per_chunk)
    print(f"Processing {len(probe_chunks)} probe chunks...")
    
    # Process each probe chunk
    for chunk_idx, probe_indices in enumerate(probe_chunks):
        print(f"  Processing chunk {chunk_idx + 1}/{len(probe_chunks)}: {len(probe_indices)} probes")
        
        # Get molecule chunk
        mol_chunk = get_molecule_probe_chunk(molecule, probe_indices)
        probe_coords = mol_chunk.probe_coords.double()
        n_probes_chunk = len(probe_coords)
        
        chunk_reconstructed = torch.zeros(n_probes_chunk, dtype=torch.float64)
        
        # Reconstruct for each atom type
        basis_offset = 0
        for atom_type in unique_types:
            type_str = str(atom_type.item())
            if type_str not in gto_dict:
                continue
                
            gto = gto_dict[type_str]
            type_mask = atom_types == atom_type
            type_coords = atom_coords[type_mask]
            n_atoms_type = len(type_coords)
            
            if n_atoms_type == 0:
                continue  # Don't increment basis_offset if no atoms of this type
            
            # Get coefficients for this atom type
            type_coeffs = coefficients[basis_offset:basis_offset + n_atoms_type * gto.outdim]
            type_coeffs = type_coeffs.view(n_atoms_type, gto.outdim)
            
            # Use scdp's forward method for efficient reconstruction
            n_probes_tensor = torch.tensor([n_probes_chunk])
            n_atoms_tensor = torch.tensor([n_atoms_type])
            
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
        
        # Store chunk results in full array
        reconstructed[probe_indices] = chunk_reconstructed
    
    return reconstructed
        

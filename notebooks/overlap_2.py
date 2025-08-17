import numpy as np
import torch
from typing import List, Tuple, Dict
import ase.data

from notebooks.mol_loader import load_single_molecule
from scdp.model.basis_set import get_basis_set, transform_basis_set, aug_etb_for_basis
from scdp.model.gtos import GTOs

def create_scdp_basis_functions(atom_types: torch.Tensor, atom_coords: torch.Tensor,
                               basis_set_name: str = 'def2-svp', use_augmentation: bool = True,
                               beta: float = 2.0) -> Tuple[Dict[int, GTOs], List[Dict]]:
    """
    Create basis functions using scdp's GTO construction methods.
    
    Args:
        atom_types: atomic numbers (N_atoms,)
        atom_coords: atomic coordinates (N_atoms, 3)
        basis_set_name: name of basis set from basis set exchange
        use_augmentation: whether to use even-tempered augmentation
        beta: parameter for even-tempered augmentation
    
    Returns:
        gto_dict: dictionary mapping atomic numbers to GTO objects
        basis_info: list of basis function information dictionaries
    """
    print(f"Creating basis functions using {basis_set_name}")
    
    # Get basis set from BSE and transform
    basis_set = transform_basis_set(get_basis_set(basis_set_name))
    
    if use_augmentation:
        print(f"Applying even-tempered augmentation with β={beta}")
        aug_basis = aug_etb_for_basis(
            basis_set,
            beta=beta,
            lmax_restriction=True,
            lmax_relax=0
        )
        # Combine original and augmented basis
        for elem in basis_set.keys():
            if elem in aug_basis:
                # Concatenate basis sets
                for key in ['Ls', 'coeffs', 'expos', 'contraction']:
                    # Adjust contraction indices for augmented basis
                    if key == 'contraction':
                        max_con = max(basis_set[elem][key]) + 1
                        aug_basis[elem][key] = [c + max_con for c in aug_basis[elem][key]]
                    basis_set[elem][key].extend(aug_basis[elem][key])
    
    # Get unique atom types
    unique_atom_types = torch.unique(atom_types[atom_types != 0])  # No virtual nodes
    
    # Create GTO objects for each element
    gto_dict = {}
    basis_info = []
    
    for z in unique_atom_types:
        z_int = int(z.item())
        if z_int in basis_set:
            # Create GTO object
            gto_dict[z_int] = GTOs(
                Ls=torch.tensor(basis_set[z_int]['Ls']),
                coeffs=torch.tensor(basis_set[z_int]['coeffs'], dtype=torch.float32),
                expos=torch.tensor(basis_set[z_int]['expos'], dtype=torch.float32),
                contraction=torch.tensor(basis_set[z_int]['contraction'], dtype=torch.float32) if basis_set[z_int]['contraction'] else None,
                normalize=True,
                cutoff=None  # No cutoff for overlap calculations
            )
            
            print(f"Element {ase.data.chemical_symbols[z_int]} (Z={z_int}): {gto_dict[z_int]}")
    
    # Create basis function info for analysis
    for i, (z, coord) in enumerate(zip(atom_types, atom_coords)):
        z_int = int(z.item())
        if z_int == 0 or z_int not in gto_dict:  # Skip virtual nodes or unsupported elements
            continue
            
        gto = gto_dict[z_int]
        element_symbol = ase.data.chemical_symbols[z_int]
        
        # Create entries for each basis function (orbital)
        for orb_idx in range(gto.outdim):
            basis_info.append({
                'atom_idx': i,
                'element': element_symbol,
                'atomic_number': z_int,
                'center': coord,
                'orbital_idx': orb_idx,
                'label': f'{element_symbol}{i}_orb{orb_idx}',
                'gto_object': gto
            })
    
    return gto_dict, basis_info

def compute_overlap_integrals_gto(molecule, gto_dict: Dict[int, GTOs], 
                                 basis_info: List[Dict]) -> Tuple[torch.Tensor, List[str]]:
    """
    Compute overlap integrals O_μ = ∫ρ(r)ω_μ(r)dr using GTO basis functions.
    
    Args:
        molecule: loaded molecular data
        gto_dict: dictionary of GTO objects by atomic number
        basis_info: list of basis function information
    
    Returns:
        overlaps: overlap integrals (N_basis,)
        labels: basis function labels
    """
    print(f"Computing overlaps for {len(basis_info)} GTO basis functions...")
    
    # Get probe coordinates and charge density
    probe_coords = molecule.probe_coords  # (N_probes, 3)
    charge_density = molecule.chg_labels  # (N_probes,)
    
    # Estimate volume element for integration
    if hasattr(molecule, 'grid_size'):
        grid_size = molecule.grid_size[0]
        cell_volume = torch.det(molecule.cell[0]).abs()
        volume_element = cell_volume / torch.prod(grid_size.float())
    else:
        cell_volume = torch.det(molecule.cell[0]).abs()
        n_probes = len(probe_coords)
        volume_element = cell_volume / n_probes
    
    print(f"Volume element: {volume_element:.6f} Ų")
    print(f"Number of probe points: {len(probe_coords)}")
    
    overlaps = []
    labels = []
    
    # Group basis functions by atomic number for efficient computation
    basis_by_element = {}
    for basis_func in basis_info:
        z = basis_func['atomic_number']
        if z not in basis_by_element:
            basis_by_element[z] = []
        basis_by_element[z].append(basis_func)
    
    for z, basis_funcs in basis_by_element.items():
        gto = gto_dict[z]
        
        # Get atom coordinates for this element
        atom_coords_z = torch.stack([bf['center'] for bf in basis_funcs])
        atom_indices = torch.tensor([bf['atom_idx'] for bf in basis_funcs])
        
        # Compute basis function values at all probe points
        # For GTO evaluation, we need to handle each atom separately
        basis_values_all = torch.zeros(len(basis_funcs), len(probe_coords))
        
        for i, basis_func in enumerate(basis_funcs):
            atom_coord = basis_func['center'].unsqueeze(0)  # (1, 3)
            orb_idx = basis_func['orbital_idx']
            
            # Use GTO forward method to compute basis function values
            n_probes = torch.tensor([len(probe_coords)])
            n_atoms = torch.tensor([1])
            
            # Get basis function values for this atom at all probe points
            basis_values = gto.forward(
                probe_coords=probe_coords,
                atom_coords=atom_coord,
                n_probes=n_probes,
                n_atoms=n_atoms,
                coeffs=None,  # Don't multiply by coefficients yet
                expo_scaling=None,
                reorder=False,  # Keep original coordinate order
                pbc=False,
                cell=None
            )  # Shape: (N_probes, gto.outdim)
            
            # Extract the specific orbital
            basis_values_all[i] = basis_values[:, orb_idx]
            
            if i % 10 == 0:
                print(f"    Evaluated {i+1}/{len(basis_funcs)} basis functions for element {ase.data.chemical_symbols[z]}")
        
        # Compute overlaps for this element
        for i, basis_func in enumerate(basis_funcs):
            overlap = torch.sum(charge_density * basis_values_all[i]) * volume_element
            overlaps.append(overlap)
            labels.append(basis_func['label'])
    
    print(f"  Completed overlap computation for all basis functions")
    
    return torch.stack(overlaps), labels

def analyze_overlaps_gto(overlaps: torch.Tensor, labels: List[str], molecule, 
                        gto_dict: Dict[int, GTOs], basis_info: List[Dict]):
    """
    Analyze and display the computed overlaps from GTO basis functions.
    
    Args:
        overlaps: computed overlap integrals
        labels: basis function labels
        molecule: molecular data for context
        gto_dict: dictionary of GTO objects
        basis_info: basis function information
    """
    print("\n" + "="*60)
    print("GTO BASIS OVERLAP ANALYSIS")
    print("="*60)
    
    print(f"Total number of electrons (sum of charge density): {molecule.chg_labels.sum():.4f}")
    print(f"Sum of all overlaps: {overlaps.sum():.4f}")
    print(f"Mean absolute overlap: {overlaps.abs().mean():.6f}")
    print(f"Standard deviation: {overlaps.std():.6f}")
    print(f"Max overlap: {overlaps.max():.6f} ({labels[overlaps.argmax()]})")
    print(f"Min overlap: {overlaps.min():.6f} ({labels[overlaps.argmin()]})")
    
    # Sort by absolute value for analysis
    abs_overlaps = overlaps.abs()
    sorted_indices = torch.argsort(abs_overlaps, descending=True)
    
    print(f"\nTop 10 largest overlaps (by absolute value):")
    print("-" * 60)
    for i in range(min(10, len(overlaps))):
        idx = sorted_indices[i]
        print(f"{labels[idx]:>20}: {overlaps[idx]:>12.6f} (|{abs_overlaps[idx]:.6f}|)")
    
    # Analyze by element
    elements = set(bf['element'] for bf in basis_info)
    for element in sorted(elements):
        element_overlaps = [overlaps[i] for i, bf in enumerate(basis_info) 
                          if bf['element'] == element]
        if element_overlaps:
            element_overlaps = torch.stack(element_overlaps)
            z = next(bf['atomic_number'] for bf in basis_info if bf['element'] == element)
            gto = gto_dict[z]
            
            print(f"\n{element} (Z={z}) orbital statistics:")
            print(f"  GTO info: {gto}")
            print(f"  Count: {len(element_overlaps)}")
            print(f"  Mean: {element_overlaps.mean():.6f}")
            print(f"  Std:  {element_overlaps.std():.6f}")
            print(f"  Sum:  {element_overlaps.sum():.6f}")
            print(f"  Percentage: {100 * element_overlaps.sum() / overlaps.sum():.1f}%")
    
    # Display basis set information
    print(f"\nBasis Set Information:")
    for z, gto in gto_dict.items():
        element = ase.data.chemical_symbols[z]
        print(f"  {element}: Lmax={gto.Lmax}, n_primitives={len(gto.Ls)}, n_contracted={gto.outdim}")

def main(basis_set_name: str = 'def2-svp', use_augmentation: bool = True, beta: float = 2.0):
    """Main function to compute and analyze overlaps using GTO basis functions."""
    print("Loading molecule...")
    molecule = load_single_molecule(idx=0)
    
    print("\n" + "="*60)
    print("CREATING GTO BASIS FUNCTIONS")
    print("="*60)
    
    # Filter out virtual nodes
    real_atoms_mask = molecule.atom_types != 0
    real_atom_types = molecule.atom_types[real_atoms_mask]
    real_atom_coords = molecule.coords[real_atoms_mask]
    
    print(f"Real atoms: {len(real_atom_types)}")
    print(f"Atom types: {real_atom_types.unique()}")
    
    # Create GTO basis functions
    gto_dict, basis_info = create_scdp_basis_functions(
        real_atom_types, real_atom_coords, 
        basis_set_name=basis_set_name,
        use_augmentation=use_augmentation,
        beta=beta
    )
    
    print(f"\nCreated basis functions for {len(gto_dict)} elements:")
    total_basis_funcs = sum(len([bf for bf in basis_info if bf['atomic_number'] == z]) for z in gto_dict.keys())
    print(f"Total basis functions: {total_basis_funcs}")
    
    print("\n" + "="*60)
    print("COMPUTING OVERLAPS")
    print("="*60)
    
    # Compute overlaps
    overlaps, labels = compute_overlap_integrals_gto(molecule, gto_dict, basis_info)
    
    # Analyze results
    analyze_overlaps_gto(overlaps, labels, molecule, gto_dict, basis_info)
    
    return overlaps, labels, gto_dict, basis_info, molecule

if __name__ == "__main__":
    # Test different basis sets
    test_configs = [
        {"basis_set_name": "sto-3g", "use_augmentation": False, "beta": 2.0},
        {"basis_set_name": "def2-svp", "use_augmentation": False, "beta": 2.0},
        {"basis_set_name": "def2-svp", "use_augmentation": True, "beta": 2.0},
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n{'='*80}")
        print(f"TESTING CONFIGURATION {i+1}: {config}")
        print(f"{'='*80}")
        
        try:
            overlaps, labels, gto_dict, basis_info, molecule = main(**config)
            
            print(f"\nSUMMARY for {config}:")
            print(f"  Total overlap sum: {overlaps.sum():.4f}")
            print(f"  Number of basis functions: {len(overlaps)}")
            print(f"  Mean absolute overlap: {overlaps.abs().mean():.6f}")
            print(f"  Electron accounting: {100 * overlaps.sum() / molecule.chg_labels.sum():.1f}%")
            
        except Exception as e:
            print(f"Configuration failed: {e}")
            continue
    
    print(f"\nGTO overlap computation completed!")

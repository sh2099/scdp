import numpy as np
import torch
from typing import List, Tuple, Dict
from scipy.special import factorial2
import ase.data

from mol_loader import load_single_molecule

def gaussian_3d(r: torch.Tensor, center: torch.Tensor, exponent: float, 
                l: int = 0, m: int = 0, n: int = 0) -> torch.Tensor:
    """
    Evaluate a 3D Gaussian basis function at positions r.
    
    Args:
        r: positions to evaluate (N, 3)
        center: center of the Gaussian (3,)
        exponent: Gaussian exponent (alpha)
        l, m, n: angular momentum quantum numbers for x^l * y^m * z^n
    
    Returns:
        Evaluated Gaussian at all positions (N,)
    """
    # Displacement from center
    dr = r - center  # (N, 3)
    x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
    
    # Distance squared
    r_sq = torch.sum(dr**2, dim=1)  # (N,)
    
    # Normalization constant for Gaussian primitive
    # N = (2*alpha/pi)^(3/4) * (4*alpha)^((l+m+n)/2) / sqrt(factorial2(2*l-1) * factorial2(2*m-1) * factorial2(2*n-1))
    norm = (2 * exponent / np.pi)**(3/4)
    if l + m + n > 0:
        norm *= (4 * exponent)**((l + m + n) / 2)
        if l > 0:
            norm /= np.sqrt(factorial2(2*l - 1))
        if m > 0:
            norm /= np.sqrt(factorial2(2*m - 1))
        if n > 0:
            norm /= np.sqrt(factorial2(2*n - 1))
    
    # Gaussian function
    gaussian = norm * (x**l) * (y**m) * (z**n) * torch.exp(-exponent * r_sq)
    
    return gaussian

def create_minimal_basis(atom_types: torch.Tensor, atom_coords: torch.Tensor) -> List[Dict]:
    """
    Create a minimal basis set for the molecule.
    Uses STO-3G like exponents for common elements.
    
    Args:
        atom_types: atomic numbers (N_atoms,)
        atom_coords: atomic coordinates (N_atoms, 3)
    
    Returns:
        List of basis function dictionaries
    """
    # STO-3G exponents for common elements (approximate)
    sto3g_exponents = {
        1: [0.168856],  # H: 1s
        6: [2.941249, 0.683483],  # C: 1s, 2s
        7: [4.173511, 0.776370],  # N: 1s, 2s  
        8: [5.695115, 0.846310],  # O: 1s, 2s
    }
    
    # p orbital exponents (approximate)
    p_exponents = {
        6: [0.222766],  # C: 2p
        7: [0.256240],  # N: 2p
        8: [0.306674],  # O: 2p
    }
    
    basis_functions = []
    
    for i, (z, coord) in enumerate(zip(atom_types, atom_coords)):
        z = int(z.item())
        if z == 0:  # Skip virtual nodes
            continue
            
        element_symbol = ase.data.chemical_symbols[z]
        
        if z in sto3g_exponents:
            # Add s orbitals
            for exp in sto3g_exponents[z]:
                basis_functions.append({
                    'atom_idx': i,
                    'element': element_symbol,
                    'center': coord,
                    'exponent': exp,
                    'l': 0, 'm': 0, 'n': 0,  # s orbital
                    'label': f'{element_symbol}{i}_s'
                })
            
            # Add p orbitals for non-hydrogen
            if z > 1 and z in p_exponents:
                for exp in p_exponents[z]:
                    # px, py, pz orbitals
                    for l, m, n, orbital in [(1,0,0,'px'), (0,1,0,'py'), (0,0,1,'pz')]:
                        basis_functions.append({
                            'atom_idx': i,
                            'element': element_symbol,
                            'center': coord,
                            'exponent': exp,
                            'l': l, 'm': m, 'n': n,
                            'label': f'{element_symbol}{i}_{orbital}'
                        })
    
    return basis_functions

def compute_overlap_integrals(molecule, basis_functions: List[Dict]) -> Tuple[torch.Tensor, List[str]]:
    """
    Compute overlap integrals O_mu = ∫ρ(r)ω_μ(r)dr using numerical integration.
    
    Args:
        molecule: loaded molecular data
        basis_functions: list of basis function definitions
    
    Returns:
        overlaps: overlap integrals (N_basis,)
        labels: basis function labels
    """
    print(f"Computing overlaps for {len(basis_functions)} basis functions...")
    
    # Get probe coordinates and charge density
    probe_coords = molecule.probe_coords  # (N_probes, 3)
    charge_density = molecule.chg_labels  # (N_probes,)
    
    # Estimate volume element for integration
    if hasattr(molecule, 'grid_size'):
        grid_size = molecule.grid_size[0]
        cell_volume = torch.det(molecule.cell[0]).abs()
        volume_element = cell_volume / torch.prod(grid_size.float())
    else:
        # Rough estimate based on probe density
        cell_volume = torch.det(molecule.cell[0]).abs()
        n_probes = len(probe_coords)
        volume_element = cell_volume / n_probes
    
    print(f"Volume element: {volume_element:.6f} Ų")
    print(f"Number of probe points: {len(probe_coords)}")
    
    overlaps = []
    labels = []
    
    for i, basis_func in enumerate(basis_functions):
        # Evaluate basis function at all probe points
        basis_values = gaussian_3d(
            probe_coords,
            basis_func['center'],
            basis_func['exponent'],
            basis_func['l'],
            basis_func['m'],
            basis_func['n']
        )
        
        # Compute overlap: O_mu = ∫ρ(r)ω_μ(r)dr ≈ Σ ρ(r_i) * ω_μ(r_i) * ΔV
        overlap = torch.sum(charge_density * basis_values) * volume_element
        overlaps.append(overlap)
        labels.append(basis_func['label'])
        
        if i % 5 == 0 or i == len(basis_functions) - 1:
            print(f"  Processed {i+1}/{len(basis_functions)} basis functions")
    
    return torch.stack(overlaps), labels

def analyze_overlaps(overlaps: torch.Tensor, labels: List[str], molecule):
    """
    Analyze and display the computed overlaps.
    
    Args:
        overlaps: computed overlap integrals
        labels: basis function labels
        molecule: molecular data for context
    """
    print("\n" + "="*60)
    print("OVERLAP ANALYSIS")
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
    print("-" * 50)
    for i in range(min(10, len(overlaps))):
        idx = sorted_indices[i]
        print(f"{labels[idx]:>15}: {overlaps[idx]:>12.6f} (|{abs_overlaps[idx]:.6f}|)")
    
    # Analyze by orbital type
    s_overlaps = [overlaps[i] for i, label in enumerate(labels) if '_s' in label]
    p_overlaps = [overlaps[i] for i, label in enumerate(labels) if '_p' in label]
    
    if s_overlaps:
        s_overlaps = torch.stack(s_overlaps)
        print(f"\nS orbital statistics:")
        print(f"  Count: {len(s_overlaps)}")
        print(f"  Mean: {s_overlaps.mean():.6f}")
        print(f"  Std:  {s_overlaps.std():.6f}")
        print(f"  Sum:  {s_overlaps.sum():.6f}")
    
    if p_overlaps:
        p_overlaps = torch.stack(p_overlaps)
        print(f"\nP orbital statistics:")
        print(f"  Count: {len(p_overlaps)}")
        print(f"  Mean: {p_overlaps.mean():.6f}")
        print(f"  Std:  {p_overlaps.std():.6f}")
        print(f"  Sum:  {p_overlaps.sum():.6f}")
    
    # Analyze by element
    elements = set(label.split('_')[0][:-1] for label in labels)  # Extract element from label
    for element in sorted(elements):
        element_overlaps = [overlaps[i] for i, label in enumerate(labels) 
                          if label.startswith(element)]
        if element_overlaps:
            element_overlaps = torch.stack(element_overlaps)
            print(f"\n{element} orbital statistics:")
            print(f"  Count: {len(element_overlaps)}")
            print(f"  Mean: {element_overlaps.mean():.6f}")
            print(f"  Sum:  {element_overlaps.sum():.6f}")

def main():
    """Main function to compute and analyze overlaps."""
    print("Loading molecule...")
    molecule = load_single_molecule(idx=0)
    
    print("\n" + "="*60)
    print("CREATING BASIS FUNCTIONS")
    print("="*60)
    
    # Filter out virtual nodes
    real_atoms_mask = molecule.atom_types != 0
    real_atom_types = molecule.atom_types[real_atoms_mask]
    real_atom_coords = molecule.coords[real_atoms_mask]
    
    print(f"Real atoms: {len(real_atom_types)}")
    print(f"Atom types: {real_atom_types.unique()}")
    
    # Create basis functions
    basis_functions = create_minimal_basis(real_atom_types, real_atom_coords)
    
    print(f"Created {len(basis_functions)} basis functions:")
    for i, basis_func in enumerate(basis_functions[:10]):  # Show first 10
        print(f"  {basis_func['label']}: α={basis_func['exponent']:.4f}, "
              f"l={basis_func['l']}, m={basis_func['m']}, n={basis_func['n']}")
    if len(basis_functions) > 10:
        print(f"  ... and {len(basis_functions) - 10} more")
    
    print("\n" + "="*60)
    print("COMPUTING OVERLAPS")
    print("="*60)
    
    # Compute overlaps
    overlaps, labels = compute_overlap_integrals(molecule, basis_functions)
    
    # Analyze results
    analyze_overlaps(overlaps, labels, molecule)
    
    return overlaps, labels, basis_functions, molecule

if __name__ == "__main__":
    overlaps, labels, basis_functions, molecule = main()
    print(f"\nOverlap computation completed successfully!")
    print(f"Results stored in variables: overlaps, labels, basis_functions, molecule")

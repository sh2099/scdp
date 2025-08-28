import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from copy import deepcopy
import random

from scdp.model.gtos import GTOs

def generate_custom_even_tempered_basis(L_values: List[int], 
                                      exponent_ranges: List[Tuple[float, float]],
                                      beta: float = 2.0,
                                      n_exponents_per_L: Optional[List[int]] = None,
                                      add_diversity: bool = False,
                                      random_seed: Optional[int] = None) -> Dict[str, Union[List, np.ndarray]]:
    """
    Generate a custom even-tempered basis set with specified L values and exponent ranges.
    
    Args:
        L_values: List of angular momentum values (e.g., [0, 1, 2] for s, p, d)
        exponent_ranges: List of (min_exp, max_exp) tuples for each L value
        beta: Geometric progression factor for even-tempered series
        n_exponents_per_L: Number of exponents per L. If None, calculated from ranges and beta
        add_diversity: If True, randomly sample starting exponent to add diversity
        random_seed: Seed for random number generation (for reproducibility)
    
    Returns:
        Dictionary with basis set format compatible with scdp
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    if len(exponent_ranges) == 1:
        exponent_ranges = exponent_ranges * len(L_values)
    elif len(L_values) != len(exponent_ranges):
        raise ValueError("L_values and exponent_ranges must have the same length")
    
    # Generate a SINGLE set of exponents that will be used for ALL L values
    # Use the first exponent range as the reference
    min_exp, max_exp = exponent_ranges[0]
    
    if min_exp >= max_exp:
        raise ValueError(f"Invalid exponent range: min_exp ({min_exp}) >= max_exp ({max_exp})")
    
    # Add extra headroom: extend upper limit by beta factor for safety
    extended_max_exp = max_exp * beta
    
    # Calculate number of exponents
    if n_exponents_per_L is None:
        n_exp = max(1, int(np.ceil(np.log(extended_max_exp / min_exp) / np.log(beta))))
    else:
        n_exp = n_exponents_per_L[0] if isinstance(n_exponents_per_L, list) else n_exponents_per_L
    
    # Determine starting exponent
    if add_diversity:
        # Create diversity by randomly sampling starting exponent
        # Range: (min_exp/beta, min_exp) to step slightly below the given minimum
        lower_bound = min_exp / beta
        starting_exp = random.uniform(lower_bound, min_exp)
    else:
        # Use the provided minimum exponent
        starting_exp = min_exp
    
    # Generate even-tempered exponents: alpha_i = starting_exp * beta^i
    shared_exponents = [starting_exp * (beta ** j) for j in range(n_exp)]
    
    # Apply the extended upper limit (beta * max_exp) instead of original max_exp
    shared_exponents = [exp for exp in shared_exponents if exp <= extended_max_exp]
    if not shared_exponents:
        shared_exponents = [starting_exp]  # Fallback to at least one exponent
    
    print(f"Generated {len(shared_exponents)} shared exponents for all L values:")
    print(f"  Original range: {min_exp:.3e} to {max_exp:.3e}")
    print(f"  Extended upper limit: {extended_max_exp:.3e} (β × max_exp)")
    print(f"  Actual range: {min(shared_exponents):.3e} to {max(shared_exponents):.3e}")
    print(f"  Exponents: {[f'{e:.3e}' for e in shared_exponents[:8]]}")
    if len(shared_exponents) > 8:
        print(f"  ... and {len(shared_exponents)-8} more")
    
    Ls = []
    coeffs = []
    expos = []
    contraction = []
    
    contraction_idx = 0
    
    # Now use the SAME exponents for ALL L values
    for L in L_values:
        for exp in shared_exponents:
            Ls.append(L)
            coeffs.append(1.0)  # Unit coefficients for primitive GTOs
            expos.append(exp)
            contraction.append(contraction_idx)
            contraction_idx += 1
    
    return {
        'Ls': Ls,
        'coeffs': coeffs,
        'expos': expos,
        'contraction': contraction
    }

def create_uniform_custom_basis(atom_types: torch.Tensor,
                               L_values: List[int],
                               exponent_ranges: List[Tuple[float, float]],
                               beta: float = 2.0,
                               n_exponents_per_L: Optional[List[int]] = None,
                               element_for_all: int = 6,
                               add_diversity: bool = False,
                               random_seed: Optional[int] = None,
                               silent: bool = False,
                               silent_analysis: bool = False) -> Dict[str, GTOs]:
    """
    Create custom GTO basis where all atoms use the same basis functions.
    
    Args:
        atom_types: Atomic numbers for all centers
        L_values: List of angular momentum values to include
        exponent_ranges: List of (min_exp, max_exp) tuples for each L value
        beta: Geometric progression factor for even-tempered series
        n_exponents_per_L: Number of exponents per L. If None, auto-calculated
        element_for_all: Atomic number to use for all atoms (default: Carbon)
        add_diversity: If True, add diversity by randomly sampling starting exponents
        random_seed: Seed for random number generation
        silent: If True, suppress print output
        silent_analysis: If True, suppress detailed analysis output
    
    Returns:
        Dictionary mapping atom type strings to GTO objects
    """
    if not silent:
        print("Creating uniform custom GTO basis")
        print(f"L values: {L_values}")
        print(f"Exponent ranges: {exponent_ranges}")
        print(f"Beta: {beta}")
        print(f"Add diversity: {add_diversity}")
        if add_diversity and random_seed is not None:
            print(f"Random seed: {random_seed}")
        print(f"Using element {element_for_all} basis for all atoms")
        print("NOTE: All L values will use the SAME set of exponents")
    
    # Generate the custom basis ONCE for the entire molecule
    custom_basis = generate_custom_even_tempered_basis(
        L_values=L_values,
        exponent_ranges=exponent_ranges,
        beta=beta,
        n_exponents_per_L=n_exponents_per_L,
        add_diversity=add_diversity,
        random_seed=random_seed
    )
    
    if not silent and not silent_analysis:
        print(f"Generated basis with {len(custom_basis['Ls'])} primitive GTOs")
        print(f"Contractions: {max(custom_basis['contraction']) + 1}")
        
        # Print detailed exponent information
        exp_array = np.array(custom_basis['expos'])
        Ls_array = np.array(custom_basis['Ls'])
        
        # Verify all L values have the same exponents
        unique_exponents = set(exp_array)
        print(f"Verification - unique exponents across all L values: {len(unique_exponents)}")
        
        print(f"Generated exponent details:")
        print(f"  Total exponents: {len(exp_array)}")
        print(f"  Unique exponents: {len(unique_exponents)}")
        print(f"  Range: {exp_array.min():.3e} to {exp_array.max():.3e}")
        print(f"  Dynamic range: {exp_array.max()/exp_array.min():.2e}")
        
        # Show exponents by L value to verify they're identical
        for L in sorted(set(Ls_array)):
            L_mask = Ls_array == L
            L_expos = exp_array[L_mask]
            L_name = ['s', 'p', 'd', 'f', 'g', 'h'][L] if L < 6 else f'L{L}'
            print(f"  {L_name} (L={L}): {len(L_expos)} exponents from {L_expos.min():.3e} to {L_expos.max():.3e}")
            
            # Verify these are the same as the first L value
            if L == sorted(set(Ls_array))[0]:
                reference_expos = sorted(L_expos)
                print(f"    Reference exponents: {[f'{e:.3e}' for e in reference_expos[:5]]}...")
            else:
                current_expos = sorted(L_expos)
                if np.allclose(reference_expos, current_expos):
                    print(f"    ✓ Identical to reference exponents")
                else:
                    print(f"    ✗ ERROR: Different from reference exponents!")
                    raise RuntimeError(f"L={L} has different exponents than reference!")
        
        if add_diversity:
            print(f"Diversity mode enabled with seed: {random_seed}")
    elif not silent:
        # Minimal output for production
        exp_array = np.array(custom_basis['expos'])
        unique_exponents = set(exp_array)
        print(f"Generated basis: {len(custom_basis['Ls'])} primitives, {len(unique_exponents)} unique exponents")
        print(f"  Range: {exp_array.min():.3e} to {exp_array.max():.3e}")
    
    # Get unique atom types and create GTO dict
    # IMPORTANT: All atom types get the SAME basis (same exponents)
    unique_atom_types = torch.unique(atom_types).tolist()
    gto_dict = {}
    
    # Create a single GTO object that will be shared by all atom types
    shared_gto = GTOs(**custom_basis, cutoff=None, normalize=True)
    
    for atom_type in unique_atom_types:
        type_str = str(atom_type)
        # All atoms use the exact same GTO object (same exponents, L values, etc.)
        gto_dict[type_str] = shared_gto
        
        if not silent:
            print(f"Element {atom_type}: using shared basis - {gto_dict[type_str]}")
    
    # Verify all atom types have identical exponents
    if not silent and not silent_analysis and len(unique_atom_types) > 1:
        print("\nVerifying exponent consistency across atom types:")
        first_expos = gto_dict[str(unique_atom_types[0])].expos
        first_Ls = gto_dict[str(unique_atom_types[0])].Ls
        
        for atom_type in unique_atom_types[1:]:
            other_expos = gto_dict[str(atom_type)].expos
            other_Ls = gto_dict[str(atom_type)].Ls
            
            expos_identical = torch.allclose(first_expos, other_expos, rtol=1e-12)
            Ls_identical = torch.equal(first_Ls, other_Ls)
            
            if expos_identical and Ls_identical:
                print(f"  ✓ Atom type {atom_type} has identical basis to type {unique_atom_types[0]}")
            else:
                print(f"  ✗ ERROR: Atom type {atom_type} has different basis!")
                if not expos_identical:
                    diff = torch.abs(first_expos - other_expos).max()
                    print(f"    Max exponent difference: {diff:.2e}")
                if not Ls_identical:
                    print(f"    L values differ")
                raise RuntimeError("Exponent mismatch detected!")
    elif not silent:
        print(f"Basis consistency verified for {len(unique_atom_types)} atom types")
    
    return gto_dict

def inspect_custom_basis(gto_dict: Dict[str, GTOs], 
                        atom_types: torch.Tensor,
                        show_details: bool = True) -> Dict[str, Dict]:
    """
    Inspect and analyze a custom GTO basis set.
    
    Args:
        gto_dict: Dictionary of GTO objects
        atom_types: Atomic numbers for centers
        show_details: Whether to print detailed information
    
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    unique_types = torch.unique(atom_types)
    
    if show_details:
        print("="*60)
        print("CUSTOM BASIS ANALYSIS")
        print("="*60)
    
    total_basis_funcs = 0
    
    # Since custom basis is uniform, analyze just one representative GTO
    representative_gto = None
    representative_type = None
    
    for atom_type in unique_types:
        type_str = str(atom_type.item())
        if type_str not in gto_dict:
            continue
            
        gto = gto_dict[type_str]
        n_atoms_of_type = int((atom_types == atom_type).sum().item())
        
        # Use first valid GTO as representative for all atom types
        if representative_gto is None:
            representative_gto = gto
            representative_type = atom_type.item()
        
        type_total = n_atoms_of_type * gto.outdim
        total_basis_funcs += type_total
        
        analysis[atom_type.item()] = {
            'n_atoms': n_atoms_of_type,
            'basis_per_atom': gto.outdim,
            'total_basis_funcs': type_total,
        }
        
        if show_details:
            print(f"Element {atom_type.item()}: {n_atoms_of_type} atoms → {type_total} basis functions")
    
    # Show detailed breakdown only once using representative GTO
    if representative_gto is not None and show_details:
        print(f"\nCustom Basis Details (uniform for all atom types):")
        
        # Extract basis information from representative
        Ls = representative_gto.Ls.cpu().numpy()
        expos = representative_gto.expos.cpu().numpy()
        coeffs = representative_gto.coeffs.cpu().numpy()
        
        # Analyze by angular momentum
        L_analysis = {}
        for L in np.unique(Ls):
            L_mask = Ls == L
            L_expos = expos[L_mask]
            L_coeffs = coeffs[L_mask]
            
            # For custom basis, each primitive becomes a contracted function
            # and each contracted function gives (2L+1) basis functions
            n_primitives = len(L_expos)
            n_basis_funcs = n_primitives * (2 * L + 1)
            
            L_analysis[int(L)] = {
                'n_primitives': n_primitives,
                'n_contracted': n_primitives,  # In custom basis, each primitive is separately contracted
                'n_basis_funcs': n_basis_funcs,
                'min_exponent': float(np.min(L_expos)),
                'max_exponent': float(np.max(L_expos)),
                'exponent_ratio': float(np.max(L_expos) / np.min(L_expos)),
                'exponents': L_expos.tolist(),
                'coefficients': L_coeffs.tolist()
            }
        
        print(f"  Primitives: {len(Ls)}, Contracted: {representative_gto.outdim}")
        print(f"  Basis functions per atom: {representative_gto.outdim}")
        print(f"  L_max: {int(np.max(Ls))}")
        print(f"  Exponent range: {np.min(expos):.2e} to {np.max(expos):.2e}")
        print(f"  Dynamic range: {np.max(expos)/np.min(expos):.2e}")
        
        print("  Per-L breakdown:")
        for L, L_info in L_analysis.items():
            L_name = ['s', 'p', 'd', 'f', 'g', 'h'][L] if L < 6 else f'L{L}'
            print(f"    {L_name} (L={L}): {L_info['n_primitives']} primitives → {L_info['n_contracted']} contracted → {L_info['n_basis_funcs']} basis funcs")
            print(f"      Exponent range: {L_info['min_exponent']:.2e} to {L_info['max_exponent']:.2e}")
        
        # Add L_analysis to all atom types for compatibility
        for atom_type in analysis.keys():
            if isinstance(analysis[atom_type], dict):
                analysis[atom_type]['L_analysis'] = L_analysis
                analysis[atom_type]['exponent_range'] = (float(np.min(expos)), float(np.max(expos)))
                analysis[atom_type]['dynamic_range'] = float(np.max(expos) / np.min(expos))
                analysis[atom_type]['n_primitives'] = len(Ls)
                analysis[atom_type]['n_contracted'] = representative_gto.outdim
                analysis[atom_type]['L_max'] = int(np.max(Ls))
    
    analysis['total_basis_functions'] = total_basis_funcs
    analysis['total_atom_types'] = len(unique_types)
    
    if show_details:
        print(f"\nTOTAL BASIS FUNCTIONS: {total_basis_funcs}")
        print(f"TOTAL ATOM TYPES: {len(unique_types)}")
    
    return analysis

def create_gto_basis_custom(atom_types: torch.Tensor, 
                           atom_coords: torch.Tensor,
                           L_values: List[int] = [0, 1, 2],
                           exponent_ranges: List[Tuple[float, float]] = [(0.1, 100.0)],
                           beta: float = 2.0,
                           n_exponents_per_L: Optional[List[int]] = None,
                           element_for_all: int = 6,
                           add_diversity: bool = False,
                           random_seed: Optional[int] = None,
                           silent: bool = False,
                           silent_analysis: bool = False) -> Dict[str, GTOs]:
    """
    Alternative create_gto_basis method using custom uniform basis.
    
    Args:
        atom_types: Atomic numbers for all centers
        atom_coords: Coordinates for all centers (not used but kept for compatibility)
        L_values: Angular momentum values to include [default: s, p, d]
        exponent_ranges: (min, max) exponent for each L [default: reasonable ranges]
        beta: Even-tempered progression factor
        n_exponents_per_L: Explicit number of exponents per L (overrides auto-calculation)
        element_for_all: Atomic number to use for basis (ignored in uniform mode)
        add_diversity: If True, add diversity by randomly sampling starting exponents
        random_seed: Seed for random number generation (for reproducibility)
        silent: Suppress output
        silent_analysis: Suppress detailed analysis output
    
    Returns:
        Dictionary mapping atom type strings to GTO objects
    """
    if not silent:
        print("="*50)
        print("CREATING CUSTOM UNIFORM GTO BASIS")
        print("="*50)
    
    gto_dict = create_uniform_custom_basis(
        atom_types=atom_types,
        L_values=L_values,
        exponent_ranges=exponent_ranges,
        beta=beta,
        n_exponents_per_L=n_exponents_per_L,
        element_for_all=element_for_all,
        add_diversity=add_diversity,
        random_seed=random_seed,
        silent=silent,
        silent_analysis=silent_analysis
    )
    
    if not silent and not silent_analysis:
        print("\nBasis creation completed!")
        inspect_custom_basis(gto_dict, atom_types, show_details=True)
    elif not silent:
        print("Basis creation completed!")
    
    return gto_dict

def validate_custom_basis_parameters(L_values: List[int],
                                    exponent_ranges: List[Tuple[float, float]],
                                    beta: float = 2.0,
                                    n_exponents_per_L: Optional[List[int]] = None,
                                    add_diversity: bool = False,
                                    random_seed: int = None) -> bool:
    """
    Validate parameters for custom basis generation.
    
    Args:
        L_values: Angular momentum values
        exponent_ranges: Exponent ranges for each L
        beta: Even-tempered factor
        n_exponents_per_L: Number of exponents per L
        add_diversity: Whether diversity is being added
    
    Returns:
        True if parameters are valid
    
    Raises:
        ValueError: If parameters are invalid
    """
    # Check L values
    if not L_values:
        raise ValueError("L_values cannot be empty")
    if any(L < 0 for L in L_values):
        raise ValueError("All L values must be non-negative")
    if len(set(L_values)) != len(L_values):
        raise ValueError("L_values must be unique")
    
    # Check exponent ranges
    if len(exponent_ranges) != len(L_values) and len(exponent_ranges) != 1:
        raise ValueError("exponent_ranges must have same length as L_values")
    
    for i, (min_exp, max_exp) in enumerate(exponent_ranges):
        if min_exp <= 0:
            raise ValueError(f"Minimum exponent for L={L_values[i]} must be positive")
        if max_exp <= min_exp:
            raise ValueError(f"Maximum exponent must be greater than minimum for L={L_values[i]}")
    
    # Check beta
    if beta <= 1.0:
        raise ValueError("Beta must be greater than 1.0")
    
    # Check n_exponents_per_L
    if n_exponents_per_L is not None:
        if isinstance(n_exponents_per_L, list) and len(n_exponents_per_L) != len(L_values):
            raise ValueError("n_exponents_per_L must have same length as L_values or be a single integer")
        if isinstance(n_exponents_per_L, list) and any(n < 1 for n in n_exponents_per_L):
            raise ValueError("All values in n_exponents_per_L must be positive")
        if isinstance(n_exponents_per_L, int) and n_exponents_per_L < 1:
            raise ValueError("n_exponents_per_L must be positive")
        
        # Warning: all L values will use the same number of exponents
        if isinstance(n_exponents_per_L, list):
            unique_n = set(n_exponents_per_L)
            if len(unique_n) > 1:
                print(f"WARNING: Different n_exponents_per_L specified {n_exponents_per_L}, but all L values will use the same exponents!")
                print(f"         Will use n_exponents_per_L[0] = {n_exponents_per_L[0]} for all L values")
    
    # Additional validation for diversity mode
    if add_diversity and beta <= 1.0:
        raise ValueError("Beta must be greater than 1.0 for diversity mode")
    
    return True

if __name__ == "__main__":
    # Example usage
    print("Testing custom GTO basis generation...")
    
    # Test parameters
    test_atom_types = torch.tensor([1, 6, 6, 8])  # H, C, C, O
    test_atom_coords = torch.randn(4, 3)
    
    # Define custom basis: s, p, d orbitals with different exponent ranges
    L_vals = [0, 1, 2]
    exp_ranges = [(8.7e-2, 5.0e+3)]
    
    # Validate parameters
    validate_custom_basis_parameters(L_vals, exp_ranges, beta=2.0)
    
    # Create custom basis
    custom_gto_dict = create_gto_basis_custom(
        atom_types=test_atom_types,
        atom_coords=test_atom_coords,
        L_values=L_vals,
        exponent_ranges=exp_ranges,
        beta=2.0,
        silent=False
    )
    
    print("\nCustom basis generation test completed!")

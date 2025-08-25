import torch
import numpy as np
import time
from typing import List, Dict, Optional
from sklearn.metrics import r2_score, mean_absolute_error

from notebooks.mol_loader_v2 import load_molecule_with_override, get_atom_centers_and_types
from notebooks.overlap_v2 import create_gto_basis, compute_overlap_integrals_scdp, compute_overlap_matrix_scdp, reconstruct_density_scdp
from scdp.model.utils import get_nmape

# Set default dtype to double precision
torch.set_default_dtype(torch.float64)

def analyze_molecule_charge_reconstruction(
    molecule_idx: int,
    regularizations: List[float],
    basis_set_name: str = 'def2-QZVPPD',
    use_augmentation: bool = True,
    beta: float = 2.0,
    use_vnodes: bool = False,
    override_atom_type: Optional[int] = None,
    vnode_elem: int = 1,
    max_probes_per_chunk: int = 50000,
    device: str = 'cpu'
) -> Dict:
    """
    Analyze charge density reconstruction for a single molecule using scdp methods.
    
    Args:
        molecule_idx: Index of molecule to analyze
        regularizations: List of regularization values to test
        basis_set_name: Basis set name
        use_augmentation: Whether to use even-tempered augmentation
        beta: Augmentation parameter
        use_vnodes: Whether to include virtual nodes
        override_atom_type: Override all atom types to this value
        vnode_elem: Element to use for virtual node basis
        max_probes_per_chunk: Maximum probes per chunk for memory efficiency
        device: Device to use for computation
        
    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*80}")
    print(f"CHARGE DENSITY RECONSTRUCTION ANALYSIS")
    print(f"Molecule: {molecule_idx}")
    print(f"Basis set: {basis_set_name}")
    print(f"Augmentation: {use_augmentation} (β={beta})")
    print(f"Virtual nodes: {use_vnodes}")
    print(f"Atom type override: {override_atom_type}")
    print(f"Max probes per chunk: {max_probes_per_chunk}")
    print(f"{'='*80}")
    
    # Step 1: Load molecule with optional atom type override
    print("\n1. Loading molecule...")
    molecule = load_molecule_with_override(
        idx=molecule_idx, 
        vnode=use_vnodes, 
        override_atom_type=override_atom_type
    )
    
    # Step 2: Get atom centers and types for basis placement
    print("\n2. Extracting atom centers and types...")
    atom_coords, atom_types, center_info = get_atom_centers_and_types(
        molecule, 
        use_vnodes=use_vnodes, 
        override_atom_type=override_atom_type
    )
    
    # Move to specified device
    device = torch.device(device)
    molecule = molecule.to(device)
    atom_coords = atom_coords.to(device)
    atom_types = atom_types.to(device)
    
    # Step 3: Construct GTO basis
    print("\n3. Constructing GTO basis...")
    gto_dict = create_gto_basis(
        atom_types=atom_types,
        atom_coords=atom_coords,
        basis_set_name=basis_set_name,
        use_augmentation=use_augmentation,
        beta=beta,
        vnode_elem=vnode_elem
    )
    
    # Calculate total basis functions (per-atom, not per-unique-type)
    total_basis_funcs = 0
    for t in torch.unique(atom_types):
        t_str = str(int(t.item()))
        if t_str in gto_dict:
            n_atoms_of_type = int((atom_types == t).sum().item())
            total_basis_funcs += n_atoms_of_type * gto_dict[t_str].outdim
    print(f"Total basis functions: {total_basis_funcs}")
    
    # Step 4: Compute overlap integrals with probe blocking
    print("\n4. Computing overlap integrals...")
    start_time = time.time()
    overlap_integrals = compute_overlap_integrals_scdp(
        molecule, gto_dict, atom_coords, atom_types, max_probes_per_chunk
    )
    overlap_time = time.time() - start_time
    print(f"Overlap computation time: {overlap_time:.2f}s")
    
    # Step 5: Compute overlap matrix with probe blocking
    print("\n5. Computing overlap matrix...")
    start_time = time.time()
    overlap_matrix = compute_overlap_matrix_scdp(
        gto_dict, atom_coords, atom_types, molecule, max_probes_per_chunk
    )
    matrix_time = time.time() - start_time
    print(f"Matrix computation time: {matrix_time:.2f}s")
    
    # Step 6: Test different regularization values
    print("\n6. Testing regularization values...")
    results = {}
    
    for reg in regularizations:
        print(f"\n  Testing regularization λ = {reg:.2e}")
        
        try:
            # Solve regularized system: (W + λI)c = b
            W_reg = overlap_matrix + reg * torch.eye(total_basis_funcs, dtype=torch.float64, device=device)
            coefficients = torch.linalg.solve(W_reg, overlap_integrals.to(device))
            
            # Reconstruct charge density with probe blocking
            print("    Reconstructing charge density...")
            start_time = time.time()
            reconstructed = reconstruct_density_scdp(
                coefficients.cpu(), gto_dict, atom_coords.cpu(), atom_types.cpu(), 
                molecule.cpu(), max_probes_per_chunk
            )
            recon_time = time.time() - start_time
            print(f"    Reconstruction time: {recon_time:.2f}s")
            
            # Compute metrics
            true_density = molecule.chg_labels.cpu()
            
            # NMAPE using scdp's method
            nmape = get_nmape(reconstructed, true_density).item()
            
            # Additional metrics
            true_np = true_density.numpy()
            recon_np = reconstructed.numpy()
            r2 = r2_score(true_np, recon_np)
            mae = mean_absolute_error(true_np, recon_np)
            mse = np.mean((true_np - recon_np)**2)
            
            results[reg] = {
                'nmape': nmape,
                'r2': r2,
                'mae': mae,
                'mse': mse,
                'coefficients': coefficients.cpu(),
                'reconstructed_density': reconstructed,
                'success': True
            }
            
            print(f"    NMAPE: {nmape:.6f}")
            print(f"    R²: {r2:.6f}")
            print(f"    MAE: {mae:.6f}")
            
        except Exception as e:
            print(f"    Failed: {e}")
            results[reg] = {
                'nmape': float('inf'),
                'r2': -float('inf'),
                'mae': float('inf'),
                'mse': float('inf'),
                'coefficients': None,
                'reconstructed_density': None,
                'success': False,
                'error': str(e)
            }
    
    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Molecule: {molecule_idx}")
    print(f"Basis functions: {total_basis_funcs}")
    print(f"Probe points: {len(molecule.chg_labels)}")
    print(f"Probe chunks used: {len(probe_chunks) if 'probe_chunks' in locals() else 'N/A'}")
    print(f"Center info: {center_info}")
    
    successful_results = {k: v for k, v in results.items() if v['success']}
    if successful_results:
        best_reg = min(successful_results.keys(), key=lambda x: successful_results[x]['nmape'])
        best_result = successful_results[best_reg]
        print(f"\nBest regularization: λ = {best_reg:.2e}")
        print(f"Best NMAPE: {best_result['nmape']:.6f}")
        print(f"Best R²: {best_result['r2']:.6f}")
    else:
        print("\nNo successful reconstructions!")
    
    return {
        'molecule_idx': molecule_idx,
        'center_info': center_info,
        'basis_info': {
            'basis_set_name': basis_set_name,
            'use_augmentation': use_augmentation,
            'beta': beta,
            'total_functions': total_basis_funcs
        },
        'results': results,
        'timings': {
            'overlap_time': overlap_time,
            'matrix_time': matrix_time
        },
        'probe_blocking': {
            'max_probes_per_chunk': max_probes_per_chunk,
            'total_probes': len(molecule.chg_labels)
        }
    }

def main():
    """Main function to demonstrate the new analysis pipeline with probe blocking."""
    
    # Test configurations
    test_configs = [
        {
            'molecule_idx': 3475,
            'regularizations': [1e-10],
            'basis_set_name': 'def2-svp',
            'use_augmentation': True,
            'use_vnodes': True,
            'override_atom_type': 6,
            'max_probes_per_chunk': 50000  # Smaller chunks for testing
        },
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n{'#'*100}")
        print(f"RUNNING TEST CONFIGURATION {i+1} WITH PROBE BLOCKING")
        print(f"{'#'*100}")
        
        try:
            result = analyze_molecule_charge_reconstruction(**config)
            
            # Print summary
            successful = sum(1 for r in result['results'].values() if r['success'])
            total = len(result['results'])
            probe_info = result['probe_blocking']
            print(f"\nConfiguration {i+1} completed: {successful}/{total} successful reconstructions")
            print(f"Processed {probe_info['total_probes']} probes in chunks of {probe_info['max_probes_per_chunk']}")
            
        except Exception as e:
            print(f"Configuration {i+1} failed: {e}")
            continue
    
    print(f"\n{'#'*100}")
    print("ALL TESTS WITH PROBE BLOCKING COMPLETED")
    print(f"{'#'*100}")

if __name__ == "__main__":
    main()

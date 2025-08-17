import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
import json
import csv
from pathlib import Path

from notebooks.overlap_2 import (
    load_single_molecule, create_scdp_basis_functions, compute_overlap_integrals_gto, 
    analyze_overlaps_gto
)
from scdp.model.utils import get_nmape

# Set default dtype to double precision
torch.set_default_dtype(torch.float64)

def get_probe_chunks(n_probes: int, max_n_probe_per_pass: int):
    """Split probe points into chunks for memory-efficient processing."""
    probe_indices = torch.arange(n_probes)
    n_pass = int(np.ceil(n_probes / max_n_probe_per_pass))
    
    probe_chunks = []
    for i in range(n_pass):
        start_idx = i * max_n_probe_per_pass
        end_idx = min((i + 1) * max_n_probe_per_pass, n_probes)
        probe_chunks.append(probe_indices[start_idx:end_idx])
    
    return n_pass, probe_chunks

def compute_overlap_integrals_gto_blocks(molecule, gto_dict: Dict[int, any], 
                                        basis_info: List[Dict], max_probes_per_block: int = 50000,
                                        device: str = 'cuda') -> Tuple[torch.Tensor, List[str]]:
    """Compute overlap integrals using block processing."""
    # Get probe coordinates and charge density with double precision
    probe_coords = molecule.probe_coords.to(device).double()
    charge_density = molecule.chg_labels.to(device).double()
    n_probes = len(probe_coords)
    
    # Estimate volume element for integration
    if hasattr(molecule, 'grid_size'):
        grid_size = molecule.grid_size[0]
        cell_volume = torch.det(molecule.cell[0]).abs().double()
        volume_element = cell_volume / torch.prod(grid_size.double())
    else:
        cell_volume = torch.det(molecule.cell[0]).abs().double()
        volume_element = cell_volume / n_probes
    
    # Initialize overlaps with double precision
    overlaps = torch.zeros(len(basis_info), device=device, dtype=torch.float64)
    labels = [bf['label'] for bf in basis_info]
    
    # Group basis functions by atomic number
    basis_by_element = {}
    for i, basis_func in enumerate(basis_info):
        z = basis_func['atomic_number']
        if z not in basis_by_element:
            basis_by_element[z] = []
        basis_by_element[z].append((i, basis_func))
    
    # Get probe chunks
    n_pass, probe_chunks = get_probe_chunks(n_probes, max_probes_per_block)
    
    # Process each block
    for pass_idx, probe_indices in enumerate(probe_chunks):
        probe_coords_block = probe_coords[probe_indices]
        charge_density_block = charge_density[probe_indices]
        
        # Process each element
        for z, basis_funcs_with_idx in basis_by_element.items():
            gto = gto_dict[z].to(device).double()
            
            for idx, basis_func in basis_funcs_with_idx:
                atom_coord = basis_func['center'].unsqueeze(0).to(device).double()
                orb_idx = basis_func['orbital_idx']
                
                n_probes_tensor = torch.tensor([len(probe_indices)], device=device)
                n_atoms_tensor = torch.tensor([1], device=device)
                
                gto_values = gto.forward(
                    probe_coords=probe_coords_block,
                    atom_coords=atom_coord,
                    n_probes=n_probes_tensor,
                    n_atoms=n_atoms_tensor,
                    coeffs=None,
                    expo_scaling=None,
                    reorder=False,
                    pbc=False,
                    cell=None
                )
                
                basis_values_block = gto_values[:, orb_idx].double()
                overlap_contribution = torch.sum(charge_density_block * basis_values_block) * volume_element.double()
                overlaps[idx] += overlap_contribution
        
        # Clear GPU memory
        del probe_coords_block, charge_density_block
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    return overlaps.cpu(), labels

def compute_overlap_matrix_gto_blocks(gto_dict: Dict, basis_info: List[Dict], molecule, 
                                     max_probes_per_block: int = 50000, device: str = 'cuda') -> torch.Tensor:
    """Compute overlap matrix using block processing."""
    n_basis = len(basis_info)
    probe_coords = molecule.probe_coords.to(device).double()
    n_probes = len(probe_coords)
    
    # Estimate volume element
    if hasattr(molecule, 'grid_size'):
        grid_size = molecule.grid_size[0]
        cell_volume = torch.det(molecule.cell[0]).abs().double()
        volume_element = cell_volume / torch.prod(grid_size.double())
    else:
        cell_volume = torch.det(molecule.cell[0]).abs().double()
        volume_element = cell_volume / n_probes
    
    # Initialize overlap matrix with double precision
    overlap_matrix = torch.zeros(n_basis, n_basis, device=device, dtype=torch.float64)
    
    # Group basis functions by atomic number
    basis_by_element = {}
    for i, basis_func in enumerate(basis_info):
        z = basis_func['atomic_number']
        if z not in basis_by_element:
            basis_by_element[z] = []
        basis_by_element[z].append((i, basis_func))
    
    # Get probe chunks
    n_pass, probe_chunks = get_probe_chunks(n_probes, max_probes_per_block)
    
    # Process each block of probes
    for pass_idx, probe_indices in enumerate(probe_chunks):
        probe_coords_block = probe_coords[probe_indices]
        basis_values_block = torch.zeros(n_basis, len(probe_indices), device=device, dtype=torch.float64)
        
        # Evaluate basis functions for this block
        for z, basis_funcs_with_idx in basis_by_element.items():
            gto = gto_dict[z].to(device).double()
            
            for idx, basis_func in basis_funcs_with_idx:
                atom_coord = basis_func['center'].unsqueeze(0).to(device).double()
                orb_idx = basis_func['orbital_idx']
                
                n_probes_tensor = torch.tensor([len(probe_indices)], device=device)
                n_atoms_tensor = torch.tensor([1], device=device)
                
                gto_values = gto.forward(
                    probe_coords=probe_coords_block,
                    atom_coords=atom_coord,
                    n_probes=n_probes_tensor,
                    n_atoms=n_atoms_tensor,
                    coeffs=None,
                    expo_scaling=None,
                    reorder=False,
                    pbc=False,
                    cell=None
                )
                
                basis_values_block[idx] = gto_values[:, orb_idx].double()
        
        # Update overlap matrix with this block's contribution
        overlap_contribution = torch.matmul(basis_values_block, basis_values_block.T) * volume_element.double()
        overlap_matrix += overlap_contribution
        
        # Clear GPU memory
        del basis_values_block, probe_coords_block, overlap_contribution
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    return overlap_matrix.cpu()

def reconstruct_charge_density_blocks(coefficients: torch.Tensor, gto_dict: Dict, basis_info: List[Dict], 
                                     molecule, max_probes_per_block: int = 50000, 
                                     device: str = 'cuda') -> torch.Tensor:
    """Reconstruct charge density using block processing."""
    probe_coords = molecule.probe_coords.to(device).double()
    coefficients = coefficients.to(device).double()
    n_probes = len(probe_coords)
    
    # Initialize result with double precision
    reconstructed_density = torch.zeros(n_probes, device=device, dtype=torch.float64)
    
    # Group basis functions by element
    basis_by_element = {}
    for i, basis_func in enumerate(basis_info):
        z = basis_func['atomic_number']
        if z not in basis_by_element:
            basis_by_element[z] = []
        basis_by_element[z].append((i, basis_func))
    
    # Get probe chunks
    n_pass, probe_chunks = get_probe_chunks(n_probes, max_probes_per_block)
    
    # Process each block
    for pass_idx, probe_indices in enumerate(probe_chunks):
        probe_coords_block = probe_coords[probe_indices]
        density_block = torch.zeros(len(probe_indices), device=device, dtype=torch.float64)
        
        # Process each element
        for z, basis_funcs_with_idx in basis_by_element.items():
            gto = gto_dict[z].to(device).double()
            
            for idx, basis_func in basis_funcs_with_idx:
                atom_coord = basis_func['center'].unsqueeze(0).to(device).double()
                orb_idx = basis_func['orbital_idx']
                
                n_probes_tensor = torch.tensor([len(probe_indices)], device=device)
                n_atoms_tensor = torch.tensor([1], device=device)
                
                gto_values = gto.forward(
                    probe_coords=probe_coords_block,
                    atom_coords=atom_coord,
                    n_probes=n_probes_tensor,
                    n_atoms=n_atoms_tensor,
                    coeffs=None,
                    expo_scaling=None,
                    reorder=False,
                    pbc=False,
                    cell=None
                )
                
                basis_values_block = gto_values[:, orb_idx].double()
                density_block += coefficients[idx].double() * basis_values_block
        
        # Store result for this block
        reconstructed_density[probe_indices] = density_block
        
        # Clear GPU memory
        del probe_coords_block, density_block
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    return reconstructed_density.cpu()

def save_molecule_results(mol_result: Dict, output_file: str):
    """
    Save molecule results to CSV file.
    
    Args:
        mol_result: Results dictionary for a single molecule
        output_file: Path to output CSV file
    """
    file_exists = Path(output_file).exists()
    
    with open(output_file, 'a', newline='') as csvfile:
        fieldnames = ['molecule_idx', 'n_basis', 'regularization', 'nmape', 'r2', 'mae']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write results for each regularization
        for reg, results in mol_result['results'].items():
            writer.writerow({
                'molecule_idx': mol_result['molecule_idx'],
                'n_basis': mol_result['n_basis'],
                'regularization': reg,
                'nmape': results['nmape'],
                'r2': results['r2'],
                'mae': results['mae']
            })

def load_results_from_file(results_file: str) -> Tuple[List[int], List[float], np.ndarray, np.ndarray]:
    """
    Load results from CSV file and organize into matrices.
    
    Args:
        results_file: Path to CSV file with results
        
    Returns:
        molecule_indices: List of molecule indices
        regularizations: List of regularization values
        nmape_matrix: NMAPE values matrix (molecules x regularizations)
        r2_matrix: R² values matrix (molecules x regularizations)
    """
    results_data = []
    with open(results_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            results_data.append({
                'molecule_idx': int(row['molecule_idx']),
                'regularization': float(row['regularization']),
                'nmape': float(row['nmape']),
                'r2': float(row['r2']),
                'mae': float(row['mae'])
            })
    
    # Extract unique molecules and regularizations
    molecule_indices = sorted(set(row['molecule_idx'] for row in results_data))
    regularizations = sorted(set(row['regularization'] for row in results_data))
    
    # Initialize matrices
    nmape_matrix = np.full((len(molecule_indices), len(regularizations)), np.nan)
    r2_matrix = np.full((len(molecule_indices), len(regularizations)), np.nan)
    
    # Fill matrices
    for row in results_data:
        mol_idx = molecule_indices.index(row['molecule_idx'])
        reg_idx = regularizations.index(row['regularization'])
        
        # Handle failed results
        if not (np.isinf(row['nmape']) or np.isnan(row['nmape'])):
            nmape_matrix[mol_idx, reg_idx] = row['nmape']
        if not (np.isinf(row['r2']) or np.isnan(row['r2']) or row['r2'] < -1e10):
            r2_matrix[mol_idx, reg_idx] = row['r2']
    
    return molecule_indices, regularizations, nmape_matrix, r2_matrix

def analyze_single_molecule(molecule_idx: int, regularizations: List[float], 
                           basis_set_name: str = 'def2-QZVPPD', 
                           use_augmentation: bool = True, beta: float = 2.0,
                           max_probes_per_block: int = 50000, device: str = 'cuda',
                           output_file: str = None) -> Dict:
    """
    Analyze a single molecule with multiple regularization values and save results.
    
    Args:
        molecule_idx: Index of molecule to analyze
        regularizations: List of regularization values to test
        basis_set_name: Basis set name
        use_augmentation: Whether to use augmentation
        beta: Beta parameter
        max_probes_per_block: Maximum probes per block
        device: Device to use
        output_file: Path to save results (if None, results not saved)
    
    Returns:
        Dictionary with results for each regularization value
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING MOLECULE {molecule_idx}")
    print(f"{'='*80}")
    
    # Load molecule and create basis functions
    molecule = load_single_molecule(idx=molecule_idx, vnode=False)
    
    # Filter out virtual nodes
    real_atoms_mask = molecule.atom_types != 0
    real_atom_types = molecule.atom_types[real_atoms_mask]
    real_atom_coords = molecule.coords[real_atoms_mask]
    
    # Create GTO basis functions
    gto_dict, basis_info = create_scdp_basis_functions(
        real_atom_types, real_atom_coords,
        basis_set_name=basis_set_name,
        use_augmentation=use_augmentation,
        beta=beta
    )
    print(f"Created {len(basis_info)} GTO basis functions")
    
    # Compute overlap integrals and matrix once
    print("Computing overlap integrals and matrix...")
    overlap_integrals, labels = compute_overlap_integrals_gto_blocks(
        molecule, gto_dict, basis_info, max_probes_per_block, device
    )
    overlap_matrix = compute_overlap_matrix_gto_blocks(
        gto_dict, basis_info, molecule, max_probes_per_block, device
    )
    
    # Test each regularization value
    results = {}
    
    for reg in regularizations:
        print(f"\n  Testing regularization {reg:.0e}...")
        
        try:
            # Move to device for solving with double precision
            overlap_matrix_gpu = overlap_matrix.to(device).double()
            overlap_integrals_gpu = overlap_integrals.to(device).double()
            
            # Add regularization
            W_reg = overlap_matrix_gpu + torch.tensor(reg, dtype=torch.float64, device=device) * torch.eye(overlap_matrix_gpu.shape[0], device=device, dtype=torch.float64)
            
            # Solve for coefficients
            coefficients = torch.linalg.solve(W_reg, overlap_integrals_gpu)
            
            # Reconstruct charge density
            reconstructed_density = reconstruct_charge_density_blocks(
                coefficients, gto_dict, basis_info, molecule, max_probes_per_block, device
            )
            
            # Compute NMAPE
            nmape = get_nmape(reconstructed_density.double(), molecule.chg_labels.double()).item()
            
            # Compute other metrics
            true_np = molecule.chg_labels.cpu().numpy()
            recon_np = reconstructed_density.cpu().numpy()
            r2 = r2_score(true_np, recon_np)
            mae = mean_absolute_error(true_np, recon_np)
            
            results[reg] = {
                'nmape': nmape,
                'r2': r2,
                'mae': mae,
                'coefficients': None,  # Don't store to save memory
                'reconstructed_density': None  # Don't store to save memory
            }
            
            print(f"    NMAPE: {nmape:.4f}, R²: {r2:.4f}")
            
        except Exception as e:
            print(f"    Failed with regularization {reg}: {e}")
            results[reg] = {
                'nmape': float('inf'),
                'r2': -float('inf'),
                'mae': float('inf'),
                'coefficients': None,
                'reconstructed_density': None
            }
    
    mol_result = {
        'molecule_idx': molecule_idx,
        'molecule': None,  # Don't store to save memory
        'results': results,
        'n_basis': len(basis_info)
    }
    
    # Save results to file if specified
    if output_file:
        save_molecule_results(mol_result, output_file)
        print(f"  Results saved to {output_file}")
    
    # Clear large objects to free memory
    del molecule, gto_dict, basis_info, overlap_integrals, overlap_matrix
    del overlap_matrix_gpu, overlap_integrals_gpu
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return mol_result

def plot_multi_molecule_analysis_from_file(results_file: str):
    """
    Create comprehensive plots from saved results file.
    
    Args:
        results_file: Path to CSV file with results
    """
    print(f"\nLoading results from {results_file}...")
    
    # Load results from file
    molecule_indices, regularizations, nmape_matrix, r2_matrix = load_results_from_file(results_file)
    
    print(f"Loaded results for {len(molecule_indices)} molecules and {len(regularizations)} regularizations")
    
    # Cap NMAPE values at 1.0 for better visualization
    nmape_matrix_capped = np.minimum(nmape_matrix, 1.0)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: NMAPE heatmap (capped at 1.0)
    im1 = axes[0, 0].imshow(nmape_matrix_capped, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1.0)
    axes[0, 0].set_xlabel('Regularization Parameter')
    axes[0, 0].set_ylabel('Molecule Index')
    axes[0, 0].set_title('NMAPE vs Regularization (capped at 1.0)')
    axes[0, 0].set_xticks(range(len(regularizations)))
    axes[0, 0].set_xticklabels([f'{reg:.0e}' for reg in regularizations], rotation=45)
    axes[0, 0].set_yticks(range(len(molecule_indices)))
    axes[0, 0].set_yticklabels([f'Mol {idx}' for idx in molecule_indices])
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=axes[0, 0])
    cbar1.set_label('NMAPE (capped at 1.0)')
    
    # Plot 2: NMAPE vs Regularization for each molecule
    for i, mol_idx in enumerate(molecule_indices):
        nmapes = nmape_matrix_capped[i, :]
        valid_mask = ~np.isnan(nmapes)
        if np.any(valid_mask):
            axes[0, 1].semilogx(np.array(regularizations)[valid_mask], nmapes[valid_mask], 
                              'o-', label=f'Mol {mol_idx}', linewidth=2, markersize=4)
    
    axes[0, 1].set_xlabel('Regularization Parameter')
    axes[0, 1].set_ylabel('NMAPE (capped at 1.0)')
    axes[0, 1].set_title('NMAPE vs Regularization by Molecule')
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: R² heatmap
    im3 = axes[1, 0].imshow(r2_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1.0)
    axes[1, 0].set_xlabel('Regularization Parameter')
    axes[1, 0].set_ylabel('Molecule Index')
    axes[1, 0].set_title('R² Score vs Regularization')
    axes[1, 0].set_xticks(range(len(regularizations)))
    axes[1, 0].set_xticklabels([f'{reg:.0e}' for reg in regularizations], rotation=45)
    axes[1, 0].set_yticks(range(len(molecule_indices)))
    axes[1, 0].set_yticklabels([f'Mol {idx}' for idx in molecule_indices])
    
    # Add colorbar
    cbar3 = plt.colorbar(im3, ax=axes[1, 0])
    cbar3.set_label('R² Score')
    
    # Plot 4: Average NMAPE vs Regularization
    mean_nmapes = np.nanmean(nmape_matrix_capped, axis=0)
    std_nmapes = np.nanstd(nmape_matrix_capped, axis=0)
    
    axes[1, 1].errorbar(regularizations, mean_nmapes, yerr=std_nmapes, 
                       fmt='o-', linewidth=2, markersize=6, capsize=5)
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xlabel('Regularization Parameter')
    axes[1, 1].set_ylabel('Average NMAPE (capped at 1.0)')
    axes[1, 1].set_title('Average NMAPE ± Std Dev')
    axes[1, 1].set_ylim(0, 1.0)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add text annotations for best average
    best_reg_idx = np.nanargmin(mean_nmapes)
    best_reg = regularizations[best_reg_idx]
    best_nmape = mean_nmapes[best_reg_idx]
    axes[1, 1].annotate(f'Best: {best_reg:.0e}\nNMAPE: {best_nmape:.4f}', 
                       xy=(best_reg, best_nmape), xytext=(10, 10),
                       textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('vn_multi_molecule_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate and print comprehensive summary
    print(f"\n{'='*80}")
    print("MULTI-MOLECULE ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nDetailed NMAPE Results:")
    print("-" * 100)
    header = f"{'Molecule':>8}"
    for reg in regularizations:
        header += f"{reg:>10.0e}"
    header += f"{'Average':>10}"
    print(header)
    print("-" * 100)
    
    molecule_averages = []
    for i, mol_idx in enumerate(molecule_indices):
        row = f"{mol_idx:>8}"
        mol_nmapes = []
        for j, reg in enumerate(regularizations):
            nmape_val = nmape_matrix[i, j]
            if not np.isnan(nmape_val) and not np.isinf(nmape_val):
                row += f"{nmape_val:>10.4f}"
                mol_nmapes.append(nmape_val)
            else:
                row += f"{'FAIL':>10}"
        
        mol_avg = np.mean(mol_nmapes) if mol_nmapes else float('inf')
        row += f"{mol_avg:>10.4f}"
        molecule_averages.append(mol_avg)
        print(row)
    
    # Regularization averages
    print("-" * 100)
    row = f"{'Average':>8}"
    reg_averages = []
    for j, reg in enumerate(regularizations):
        reg_nmapes = [nmape_matrix[i, j] for i in range(len(molecule_indices)) 
                     if not np.isnan(nmape_matrix[i, j]) and not np.isinf(nmape_matrix[i, j])]
        reg_avg = np.mean(reg_nmapes) if reg_nmapes else float('inf')
        row += f"{reg_avg:>10.4f}"
        reg_averages.append(reg_avg)
    
    overall_avg = np.mean([avg for avg in molecule_averages if not np.isinf(avg)])
    row += f"{overall_avg:>10.4f}"
    print(row)
    print("-" * 100)
    
    # Best results
    best_mol_idx = np.argmin(molecule_averages)
    best_reg_idx = np.nanargmin(reg_averages)
    
    print(f"\nSummary Statistics:")
    print(f"  Overall average NMAPE: {overall_avg:.4f}")
    print(f"  Best molecule: Mol {molecule_indices[best_mol_idx]} (avg NMAPE: {molecule_averages[best_mol_idx]:.4f})")
    print(f"  Best regularization: {regularizations[best_reg_idx]:.0e} (avg NMAPE: {reg_averages[best_reg_idx]:.4f})")
    print(f"  Best single result: {np.nanmin(nmape_matrix):.4f}")
    
    # Find best single result
    min_idx = np.unravel_index(np.nanargmin(nmape_matrix), nmape_matrix.shape)
    best_mol = molecule_indices[min_idx[0]]
    best_reg_single = regularizations[min_idx[1]]
    print(f"  Best single result: Mol {best_mol} with reg {best_reg_single:.0e}")
    
    print(f"\nPlot saved as 'vn_multi_molecule_analysis.png'")
    
    # Save summary statistics to JSON
    summary = {
        'nmape_matrix': nmape_matrix.tolist(),
        'molecule_averages': molecule_averages,
        'regularization_averages': reg_averages,
        'overall_average': overall_avg,
        'best_molecule': molecule_indices[best_mol_idx],
        'best_regularization': regularizations[best_reg_idx],
        'molecule_indices': molecule_indices,
        'regularizations': regularizations
    }
    
    summary_file = results_file.replace('.csv', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary statistics saved to {summary_file}")
    
    return summary

def main():
    """Main function to run multi-molecule analysis."""
    print("="*80)
    print("MULTI-MOLECULE REGULARIZATION ANALYSIS")
    print("="*80)
    print("Using def2-QZVPPD basis set with block processing")
    print("Max probes per block: 50000")
    print("Using double precision (float64)")
    
    # Configuration
    molecule_indices = [5]
    print(f"Selected {len(molecule_indices)} molecules for analysis:")
    print(molecule_indices)
    regularizations = [1e-12, 1e-10, 1e-8]
    max_probes_per_block = 50000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Output file for results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"vn_multi_molecule_results_{timestamp}.csv"
    
    print(f"Analyzing molecules: {molecule_indices}")
    print(f"Testing regularizations: {regularizations}")
    print(f"Using device: {device}")
    print(f"Results will be saved to: {results_file}")
    
    # Analyze each molecule
    total_start_time = time.time()
    
    for i, mol_idx in enumerate(molecule_indices):
        mol_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"PROCESSING MOLECULE {i+1}/{len(molecule_indices)}")
        print(f"{'='*60}")
        
        try:
            mol_results = analyze_single_molecule(
                mol_idx, regularizations,
                basis_set_name='def2-QZVPPD',
                use_augmentation=True,
                beta=2.0,
                max_probes_per_block=max_probes_per_block,
                device=device,
                output_file=results_file
            )
            
            mol_time = time.time() - mol_start_time
            print(f"Completed molecule {mol_idx} in {mol_time:.1f}s")
            
            # Free memory
            del mol_results
            
        except Exception as e:
            print(f"Failed to process molecule {mol_idx}: {e}")
            continue
    
    total_time = time.time() - total_start_time
    print(f"\nTotal analysis time: {total_time:.1f}s")
    
    # Create comprehensive analysis from saved file
    print(f"\n{'='*80}")
    print("CREATING ANALYSIS FROM SAVED RESULTS")
    print(f"{'='*80}")
    
    summary = plot_multi_molecule_analysis_from_file(results_file)
    
    print(f"\nAnalysis completed successfully!")
    print(f"Results saved to: {results_file}")
    print(f"Summary statistics available in 'summary' variable")
    
    return results_file, summary

if __name__ == "__main__":
    results_file, summary = main()

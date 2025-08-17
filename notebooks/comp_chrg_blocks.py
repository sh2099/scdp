import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from copy import deepcopy
import time

from notebooks.overlap_2 import (
    load_single_molecule, create_scdp_basis_functions, compute_overlap_integrals_gto, 
    analyze_overlaps_gto
)
from scdp.model.utils import get_nmape

# Set default dtype to double precision
torch.set_default_dtype(torch.float64)

def get_probe_chunks(n_probes: int, max_n_probe_per_pass: int):
    """
    Split probe points into chunks for memory-efficient processing.
    Adapted from test.py
    
    Args:
        n_probes: total number of probe points
        max_n_probe_per_pass: maximum number of probes to process at once
        
    Returns:
        n_pass: number of passes needed
        probe_chunks: list of probe indices for each pass
    """
    probe_indices = torch.arange(n_probes)
    n_pass = int(np.ceil(n_probes / max_n_probe_per_pass))
    
    probe_chunks = []
    for i in range(n_pass):
        start_idx = i * max_n_probe_per_pass
        end_idx = min((i + 1) * max_n_probe_per_pass, n_probes)
        probe_chunks.append(probe_indices[start_idx:end_idx])
    
    return n_pass, probe_chunks

def compute_overlap_matrix_gto_blocks(gto_dict: Dict, basis_info: List[Dict], molecule, 
                                     max_probes_per_block: int = 50000, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the overlap matrix W_μν = ∫ω_μ(r)ω_ν(r)dr using GTO basis functions with block processing.
    
    Args:
        gto_dict: dictionary of GTO objects by atomic number
        basis_info: list of basis function information
        molecule: molecular data for probe coordinates and volume element
        max_probes_per_block: maximum number of probes to process per block
        device: device to use for computation
    
    Returns:
        overlap_matrix: W_μν matrix (N_basis, N_basis)
        basis_values: evaluated basis function values (N_basis, N_probes)
    """
    n_basis = len(basis_info)
    probe_coords = molecule.probe_coords.to(device).double()  # Ensure double precision
    n_probes = len(probe_coords)
    
    print(f"Computing {n_basis}x{n_basis} overlap matrix using GTO basis functions with blocks...")
    print(f"Processing {n_probes} probes in blocks of {max_probes_per_block}")
    print(f"Using double precision (float64)")
    
    # Estimate volume element
    if hasattr(molecule, 'grid_size'):
        grid_size = molecule.grid_size[0]
        cell_volume = torch.det(molecule.cell[0]).abs().double()
        volume_element = cell_volume / torch.prod(grid_size.double())
    else:
        cell_volume = torch.det(molecule.cell[0]).abs().double()
        volume_element = cell_volume / n_probes
    
    # Initialize overlap matrix and basis values storage with double precision
    overlap_matrix = torch.zeros(n_basis, n_basis, device=device, dtype=torch.float64)
    basis_values = torch.zeros(n_basis, n_probes, device=device, dtype=torch.float64)
    
    # Group basis functions by atomic number for efficient computation
    basis_by_element = {}
    for i, basis_func in enumerate(basis_info):
        z = basis_func['atomic_number']
        if z not in basis_by_element:
            basis_by_element[z] = []
        basis_by_element[z].append((i, basis_func))
    
    # Get probe chunks
    n_pass, probe_chunks = get_probe_chunks(n_probes, max_probes_per_block)
    
    print(f"  Evaluating all GTO basis functions in {n_pass} passes...")
    
    # Process each block of probes
    for pass_idx, probe_indices in enumerate(probe_chunks):
        print(f"    Processing block {pass_idx + 1}/{n_pass} ({len(probe_indices)} probes)...")
        
        probe_coords_block = probe_coords[probe_indices]
        basis_values_block = torch.zeros(n_basis, len(probe_indices), device=device, dtype=torch.float64)
        
        # Evaluate basis functions for this block
        for z, basis_funcs_with_idx in basis_by_element.items():
            gto = gto_dict[z].to(device).double()  # Ensure GTO is in double precision
            
            for idx, basis_func in basis_funcs_with_idx:
                atom_coord = basis_func['center'].unsqueeze(0).to(device).double()
                orb_idx = basis_func['orbital_idx']
                
                # Use GTO forward method to compute basis function values
                n_probes_tensor = torch.tensor([len(probe_indices)], device=device)
                n_atoms_tensor = torch.tensor([1], device=device)
                
                # Get basis function values for this atom at block probe points
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
                
                # Extract the specific orbital and ensure double precision
                basis_values_block[idx] = gto_values[:, orb_idx].double()
        
        # Store basis values for this block
        basis_values[:, probe_indices] = basis_values_block
        
        # Update overlap matrix with this block's contribution
        # W_μν += Σ_block ω_μ(r_i) * ω_ν(r_i) * ΔV
        overlap_contribution = torch.matmul(basis_values_block, basis_values_block.T) * volume_element.double()
        overlap_matrix += overlap_contribution
        
        # Clear GPU memory
        del basis_values_block, probe_coords_block, overlap_contribution
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    print("  Overlap matrix computation completed")
    
    return overlap_matrix.cpu(), basis_values.cpu()

def compute_overlap_integrals_gto_blocks(molecule, gto_dict: Dict[int, any], 
                                        basis_info: List[Dict], max_probes_per_block: int = 50000,
                                        device: str = 'cuda') -> Tuple[torch.Tensor, List[str]]:
    """
    Compute overlap integrals O_mu = ∫ρ(r)ω_μ(r)dr using GTO basis functions with block processing.
    
    Args:
        molecule: loaded molecular data
        gto_dict: dictionary of GTO objects by atomic number
        basis_info: list of basis function information
        max_probes_per_block: maximum number of probes to process per block
        device: device to use for computation
    
    Returns:
        overlaps: overlap integrals (N_basis,)
        labels: basis function labels
    """
    print(f"Computing overlaps for {len(basis_info)} GTO basis functions with blocks...")
    print(f"Using double precision (float64)")
    
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
    
    print(f"Volume element: {volume_element:.6f} Ų")
    print(f"Number of probe points: {n_probes}")
    
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
    
    print(f"Processing in {n_pass} blocks...")
    
    # Process each block
    for pass_idx, probe_indices in enumerate(probe_chunks):
        print(f"  Processing block {pass_idx + 1}/{n_pass} ({len(probe_indices)} probes)...")
        
        probe_coords_block = probe_coords[probe_indices]
        charge_density_block = charge_density[probe_indices]
        
        # Process each element
        for z, basis_funcs_with_idx in basis_by_element.items():
            gto = gto_dict[z].to(device).double()  # Ensure double precision
            
            for idx, basis_func in basis_funcs_with_idx:
                atom_coord = basis_func['center'].unsqueeze(0).to(device).double()
                orb_idx = basis_func['orbital_idx']
                
                # Use GTO forward method
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
                
                # Extract orbital and compute overlap contribution with double precision
                basis_values_block = gto_values[:, orb_idx].double()
                overlap_contribution = torch.sum(charge_density_block * basis_values_block) * volume_element.double()
                overlaps[idx] += overlap_contribution
        
        # Clear GPU memory
        del probe_coords_block, charge_density_block
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"  Completed overlap computation for all basis functions")
    
    return overlaps.cpu(), labels

def reconstruct_charge_density_blocks(coefficients: torch.Tensor, gto_dict: Dict, basis_info: List[Dict], 
                                     molecule, max_probes_per_block: int = 50000, 
                                     device: str = 'cuda') -> torch.Tensor:
    """
    Reconstruct charge density using ρ(r) = Σ c_ν * ω_ν(r) with block processing.
    
    Args:
        coefficients: c_ν vector (N_basis,)
        gto_dict: dictionary of GTO objects by atomic number
        basis_info: list of basis function information
        molecule: molecular data
        max_probes_per_block: maximum number of probes to process per block
        device: device to use for computation
    
    Returns:
        reconstructed_density: ρ(r) vector (N_probes,)
    """
    print("Reconstructing charge density with block processing...")
    print(f"Using double precision (float64)")
    
    probe_coords = molecule.probe_coords.to(device).double()
    coefficients = coefficients.to(device).double()  # Ensure double precision
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
    
    print(f"Processing reconstruction in {n_pass} blocks...")
    
    # Process each block
    for pass_idx, probe_indices in enumerate(probe_chunks):
        print(f"  Processing block {pass_idx + 1}/{n_pass} ({len(probe_indices)} probes)...")
        
        probe_coords_block = probe_coords[probe_indices]
        density_block = torch.zeros(len(probe_indices), device=device, dtype=torch.float64)
        
        # Process each element
        for z, basis_funcs_with_idx in basis_by_element.items():
            gto = gto_dict[z].to(device).double()  # Ensure double precision
            
            for idx, basis_func in basis_funcs_with_idx:
                atom_coord = basis_func['center'].unsqueeze(0).to(device).double()
                orb_idx = basis_func['orbital_idx']
                
                # Use GTO forward method
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
                
                # Add contribution: ρ(r) += c_ν * ω_ν(r) with double precision
                basis_values_block = gto_values[:, orb_idx].double()
                density_block += coefficients[idx].double() * basis_values_block
        
        # Store result for this block
        reconstructed_density[probe_indices] = density_block
        
        # Clear GPU memory
        del probe_coords_block, density_block
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    return reconstructed_density.cpu()

def analyze_reconstruction(true_density: torch.Tensor, reconstructed_density: torch.Tensor, 
                         coefficients: torch.Tensor, labels: List[str], molecule):
    """
    Analyze the quality of charge density reconstruction.
    """
    print("\n" + "="*60)
    print("BLOCK-BASED GTO RECONSTRUCTION ANALYSIS")
    print("="*60)
    
    # Estimate volume element for proper electron counting
    if hasattr(molecule, 'grid_size'):
        grid_size = molecule.grid_size[0]
        cell_volume = torch.det(molecule.cell[0]).abs()
        volume_element = cell_volume / torch.prod(grid_size.float())
    else:
        cell_volume = torch.det(molecule.cell[0]).abs()
        n_probes = len(molecule.probe_coords)
        volume_element = cell_volume / n_probes
    
    # Convert to numpy for sklearn metrics
    true_np = true_density.cpu().numpy()
    recon_np = reconstructed_density.cpu().numpy()
    volume_element_np = volume_element.cpu().numpy() if isinstance(volume_element, torch.Tensor) else volume_element
    
    # Compute metrics
    r2 = r2_score(true_np, recon_np)
    mae = mean_absolute_error(true_np, recon_np)
    mse = mean_squared_error(true_np, recon_np)
    rmse = np.sqrt(mse)
    
    # Use tensors for get_nmape function
    nmape = get_nmape(reconstructed_density, true_density).item()

    # Proper electron counting
    n_electrons_true = np.sum(true_np) * volume_element_np
    n_electrons_recon = np.sum(recon_np) * volume_element_np
    mae_per_electron = mae / n_electrons_true if n_electrons_true > 0 else float('inf')
    nmape_per_electron = nmape / n_electrons_true if n_electrons_true > 0 else float('inf')
    
    # Relative errors
    mean_true = np.mean(np.abs(true_np))
    relative_mae = mae / mean_true
    relative_rmse = rmse / mean_true
    
    print(f"Block Processing Reconstruction Quality Metrics:")
    print(f"  R² score: {r2:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  MAE per electron: {mae_per_electron:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  NMAPE: {nmape:.6f}")
    print(f"  NMAPE per electron: {nmape_per_electron:.6f}")
    print(f"  Relative MAE: {relative_mae:.2%}")
    print(f"  Relative RMSE: {relative_rmse:.2%}")
    
    print(f"\nElectron counting (with volume element):")
    print(f"  Volume element: {volume_element_np:.6f} Ų")
    print(f"  Total electrons (true): {n_electrons_true:.4f}")
    print(f"  Total electrons (reconstructed): {n_electrons_recon:.4f}")
    print(f"  Electron difference: {n_electrons_recon - n_electrons_true:.4f}")
    print(f"  Electron conservation error: {(n_electrons_recon - n_electrons_true) / n_electrons_true:.2%}")
    
    return r2, mae, rmse, relative_mae, relative_rmse, mae_per_electron, nmape

def main_blocks(molecule_idx: int = 0, regularization: float = 1e-8, create_plots: bool = True,
               basis_set_name: str = 'def2-svp', use_augmentation: bool = True, beta: float = 2.0,
               max_probes_per_block: int = 50000, device: str = 'cuda'):
    """
    Main function to perform charge density reconstruction using GTO basis functions with block processing.
    
    Args:
        molecule_idx: index of molecule to analyze
        regularization: regularization parameter for solving linear system
        create_plots: whether to create comparison plots
        basis_set_name: name of basis set to use
        use_augmentation: whether to use even-tempered augmentation
        beta: parameter for even-tempered augmentation
        max_probes_per_block: maximum number of probes to process per block
        device: device to use for computation ('cuda' or 'cpu')
    """
    print("="*60)
    print("BLOCK-BASED GTO CHARGE DENSITY RECONSTRUCTION")
    print("="*60)
    print(f"Using device: {device}")
    print(f"Max probes per block: {max_probes_per_block}")
    print(f"Using double precision (float64)")
    
    # Initialize timing dictionary
    timing = {}
    total_start_time = time.time()
    
    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Load molecule
    step_start = time.time()
    print("Step 1: Loading molecule and creating GTO basis functions...")
    molecule = load_single_molecule(idx=molecule_idx)
    
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
    timing['step1_load_and_create_basis'] = time.time() - step_start
    
    # Compute overlap integrals using blocks
    step_start = time.time()
    print(f"\nStep 2: Computing overlap integrals with block processing...")
    overlap_integrals, labels = compute_overlap_integrals_gto_blocks(
        molecule, gto_dict, basis_info, max_probes_per_block, device
    )
    timing['step2_overlap_integrals'] = time.time() - step_start
    
    step_start = time.time()
    print(f"\nStep 3: Computing overlap matrix with block processing...")
    overlap_matrix, _ = compute_overlap_matrix_gto_blocks(
        gto_dict, basis_info, molecule, max_probes_per_block, device
    )
    timing['step3_overlap_matrix'] = time.time() - step_start
    
    step_start = time.time()
    print(f"\nStep 4: Solving for coefficients c_ν...")
    # Move to device for solving with double precision
    overlap_matrix = overlap_matrix.to(device).double()
    overlap_integrals = overlap_integrals.to(device).double()
    
    # Add regularization with double precision
    W_reg = overlap_matrix + torch.tensor(regularization, dtype=torch.float64, device=device) * torch.eye(overlap_matrix.shape[0], device=device, dtype=torch.float64)
    
    print(f"  Overlap matrix condition number: {torch.linalg.cond(overlap_matrix):.2e}")
    print(f"  Regularized matrix condition number: {torch.linalg.cond(W_reg):.2e}")
    
    # Solve with double precision
    try:
        coefficients = torch.linalg.solve(W_reg, overlap_integrals)
        print("  Successfully solved using direct method")
    except:
        print("  Direct solve failed, using least squares...")
        coefficients = torch.linalg.lstsq(W_reg, overlap_integrals)[0]
    timing['step4_solve_coefficients'] = time.time() - step_start
    
    step_start = time.time()
    print(f"\nStep 5: Reconstructing charge density with block processing...")
    reconstructed_density = reconstruct_charge_density_blocks(
        coefficients, gto_dict, basis_info, molecule, max_probes_per_block, device
    )
    timing['step5_reconstruct_density'] = time.time() - step_start
    
    step_start = time.time()
    print(f"\nStep 6: Analyzing reconstruction quality...")
    metrics = analyze_reconstruction(
        molecule.chg_labels, reconstructed_density, coefficients.cpu(), labels, molecule
    )
    timing['step6_analysis'] = time.time() - step_start
    
    if create_plots:
        step_start = time.time()
        print(f"\nStep 7: Creating comparison plots...")
        # Import plotting functions from comp_chrg_2
        from notebooks.comp_chrg_2 import plot_reconstruction_comparison
        plot_reconstruction_comparison(
            molecule.chg_labels, reconstructed_density, molecule
        )
        timing['step7_plotting'] = time.time() - step_start
    
    timing['total_time'] = time.time() - total_start_time
    
    # Print timing summary
    print(f"\n{'='*60}")
    print("TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"Step 1 - Load & Create Basis: {timing['step1_load_and_create_basis']:.2f}s")
    print(f"Step 2 - Overlap Integrals:   {timing['step2_overlap_integrals']:.2f}s")
    print(f"Step 3 - Overlap Matrix:      {timing['step3_overlap_matrix']:.2f}s")
    print(f"Step 4 - Solve Coefficients:  {timing['step4_solve_coefficients']:.2f}s")
    print(f"Step 5 - Reconstruct Density: {timing['step5_reconstruct_density']:.2f}s")
    print(f"Step 6 - Analysis:            {timing['step6_analysis']:.2f}s")
    if 'step7_plotting' in timing:
        print(f"Step 7 - Plotting:            {timing['step7_plotting']:.2f}s")
    print(f"Total Time:                   {timing['total_time']:.2f}s")
    
    # Calculate computational vs overhead time
    computational_time = (timing['step2_overlap_integrals'] + 
                         timing['step3_overlap_matrix'] + 
                         timing['step5_reconstruct_density'])
    overhead_time = timing['total_time'] - computational_time
    
    print(f"\nComputational time (Steps 2,3,5): {computational_time:.2f}s ({100*computational_time/timing['total_time']:.1f}%)")
    print(f"Overhead time (other steps):      {overhead_time:.2f}s ({100*overhead_time/timing['total_time']:.1f}%)")
    
    # Return results
    results = {
        'molecule': molecule,
        'gto_dict': gto_dict,
        'basis_info': basis_info,
        'overlap_integrals': overlap_integrals.cpu().double(),
        'overlap_matrix': overlap_matrix.cpu().double(),
        'coefficients': coefficients.cpu().double(),
        'reconstructed_density': reconstructed_density.double(),
        'true_density': molecule.chg_labels.double(),
        'labels': labels,
        'basis_set_name': basis_set_name,
        'use_augmentation': use_augmentation,
        'beta': beta,
        'max_probes_per_block': max_probes_per_block,
        'device': device,
        'timing': timing,
        'metrics': {
            'r2': metrics[0],
            'mae': metrics[1], 
            'rmse': metrics[2],
            'relative_mae': metrics[3],
            'relative_rmse': metrics[4],
            'mae_per_electron': metrics[5],
            'nmape': metrics[6]
        }
    }
    
    print(f"\nBlock-based GTO charge density reconstruction completed!")
    print(f"Basis set: {basis_set_name}, Augmentation: {use_augmentation}")
    print(f"Block size: {max_probes_per_block}, Device: {device}")
    print(f"R² = {metrics[0]:.4f}, Relative RMSE = {metrics[4]:.2%}, MAE/electron = {metrics[5]:.6f}")
    
    return results

def plot_timing_comparison(results_list: List[Dict]):
    """
    Create plots comparing timing across different configurations.
    
    Args:
        results_list: List of results dictionaries with timing information
    """
    print("\nCreating timing comparison plots...")
    
    if len(results_list) < 2:
        print("Need at least 2 results to create timing comparison")
        return
    
    # Extract timing data
    config_names = []
    step_times = {
        'load_basis': [],
        'overlap_integrals': [],
        'overlap_matrix': [],
        'solve_coeffs': [],
        'reconstruct': [],
        'analysis': [],
        'total': []
    }
    
    for result in results_list:
        config_name = f"{result['max_probes_per_block']}k-{result['device']}"
        config_names.append(config_name)
        
        timing = result['timing']
        step_times['load_basis'].append(timing['step1_load_and_create_basis'])
        step_times['overlap_integrals'].append(timing['step2_overlap_integrals'])
        step_times['overlap_matrix'].append(timing['step3_overlap_matrix'])
        step_times['solve_coeffs'].append(timing['step4_solve_coefficients'])
        step_times['reconstruct'].append(timing['step5_reconstruct_density'])
        step_times['analysis'].append(timing['step6_analysis'])
        step_times['total'].append(timing['total_time'])
    
    # Create timing comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Total time comparison
    bars = axes[0, 0].bar(config_names, step_times['total'], color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Total Execution Time')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, step_times['total']):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + max(step_times['total'])*0.01,
                       f'{value:.1f}s', ha='center', va='bottom')
    
    # Plot 2: Computational steps comparison
    x = np.arange(len(config_names))
    width = 0.25
    
    bars1 = axes[0, 1].bar(x - width, step_times['overlap_integrals'], width, 
                          label='Overlap Integrals', alpha=0.7)
    bars2 = axes[0, 1].bar(x, step_times['overlap_matrix'], width, 
                          label='Overlap Matrix', alpha=0.7)
    bars3 = axes[0, 1].bar(x + width, step_times['reconstruct'], width, 
                          label='Reconstruction', alpha=0.7)
    
    axes[0, 1].set_title('Computational Steps Timing')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(config_names, rotation=45)
    axes[0, 1].legend()
    
    # Plot 3: Stacked bar chart of all steps
    bottom = np.zeros(len(config_names))
    step_labels = ['Load/Basis', 'Overlap Int.', 'Overlap Mat.', 'Solve', 'Reconstruct', 'Analysis']
    step_data = [step_times['load_basis'], step_times['overlap_integrals'], 
                step_times['overlap_matrix'], step_times['solve_coeffs'],
                step_times['reconstruct'], step_times['analysis']]
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF']
    
    for i, (data, label, color) in enumerate(zip(step_data, step_labels, colors)):
        axes[1, 0].bar(config_names, data, bottom=bottom, label=label, color=color, alpha=0.7)
        bottom += np.array(data)
    
    axes[1, 0].set_title('Breakdown of Execution Time')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 4: Performance metrics vs block size
    if all('cuda' in name or 'cpu' in name for name in config_names):
        block_sizes = [int(name.split('k-')[0]) for name in config_names]
        r2_scores = [result['metrics']['r2'] for result in results_list]
        
        ax4_twin = axes[1, 1].twinx()
        
        line1 = axes[1, 1].plot(block_sizes, step_times['total'], 'bo-', label='Total Time', linewidth=2)
        line2 = ax4_twin.plot(block_sizes, r2_scores, 'ro-', label='R² Score', linewidth=2)
        
        axes[1, 1].set_xlabel('Block Size (thousands)')
        axes[1, 1].set_ylabel('Time (seconds)', color='b')
        ax4_twin.set_ylabel('R² Score', color='r')
        axes[1, 1].set_title('Performance vs Block Size')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[1, 1].legend(lines, labels, loc='center left')
    else:
        axes[1, 1].text(0.5, 0.5, 'Mixed device types\nCannot plot vs block size', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Performance vs Block Size')
    
    plt.tight_layout()
    plt.savefig('timing_comparison_blocks.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print timing summary table
    print(f"\nTiming Comparison Summary:")
    print("-" * 80)
    print(f"{'Config':>15} {'Total(s)':>8} {'Overlap Int':>11} {'Overlap Mat':>11} {'Reconstruct':>11} {'R²':>8}")
    print("-" * 80)
    
    for i, (config, result) in enumerate(zip(config_names, results_list)):
        timing = result['timing']
        r2 = result['metrics']['r2']
        print(f"{config:>15} {timing['total_time']:>8.1f} {timing['step2_overlap_integrals']:>11.1f} "
              f"{timing['step3_overlap_matrix']:>11.1f} {timing['step5_reconstruct_density']:>11.1f} "
              f"{r2:>8.4f}")
    
    print("-" * 80)
    print(f"Plot saved as 'timing_comparison_blocks.png'")

if __name__ == "__main__":
    # Test block-based reconstruction
    print("="*80)
    print("BLOCK-BASED RECONSTRUCTION ANALYSIS")
    print("="*80)
    print("Using double precision (float64) for all computations")
    
    # Test different block sizes and devices
    test_configs = [
        {"max_probes_per_block": 20000, "device": "cuda" if torch.cuda.is_available() else "cpu"},
        {"max_probes_per_block": 40000, "device": "cuda" if torch.cuda.is_available() else "cpu"},
        {"max_probes_per_block": 60000, "device": "cuda" if torch.cuda.is_available() else "cpu"},
        {"max_probes_per_block": 80000, "device": "cuda" if torch.cuda.is_available() else "cpu"},
        #{"max_probes_per_block": 200000, "device": "cuda" if torch.cuda.is_available() else "cpu"},
    ]
    
    mol_idx = 18
    basis_config = {
        "basis_set_name": "def2-QZVPPD",
        "use_augmentation": True,
        "beta": 2.0,
        "regularization": 1e-6
    }
    
    best_results = None
    best_r2 = -np.inf
    all_results = []
    
    for i, config in enumerate(test_configs):
        print(f"\n{'='*60}")
        print(f"TESTING BLOCK CONFIGURATION {i+1}: {config}")
        print(f"{'='*60}")
        
        try:
            results = main_blocks(
                molecule_idx=mol_idx,
                create_plots=False,
                **basis_config,
                **config
            )
            
            current_r2 = results['metrics']['r2']
            all_results.append(results)
            
            if current_r2 > best_r2:
                best_r2 = current_r2
                best_results = results
                
            print(f"Configuration summary:")
            print(f"  R² = {current_r2:.4f}")
            print(f"  NMAPE = {results['metrics']['nmape']:.4f}")
            print(f"  MAE/electron = {results['metrics']['mae_per_electron']:.6f}")
            print(f"  Total time = {results['timing']['total_time']:.2f}s")
            
        except Exception as e:
            print(f"Configuration failed: {e}")
            continue
    
    # Create timing comparison plots
    if len(all_results) > 1:
        plot_timing_comparison(all_results)
    
    if best_results is not None:
        print(f"\n{'='*80}")
        print(f"BEST BLOCK CONFIGURATION RESULTS")
        print(f"{'='*80}")
        print(f"Best R² = {best_r2:.4f}")
        print(f"Best block size: {best_results['max_probes_per_block']}")
        print(f"Best device: {best_results['device']}")
        print(f"NMAPE = {best_results['metrics']['nmape']:.4f}")
        print(f"MAE/electron = {best_results['metrics']['mae_per_electron']:.6f}")
        print(f"Total time = {best_results['timing']['total_time']:.2f}s")
        
        # Create plots for best configuration
        from notebooks.comp_chrg_2 import plot_reconstruction_comparison
        plot_reconstruction_comparison(
            best_results['true_density'], 
            best_results['reconstructed_density'], 
            best_results['molecule']
        )
    
    print(f"\nBlock-based reconstruction analysis completed!")
    print(f"Results stored in 'best_results' and 'all_results' variables")

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from overlap import (
    load_single_molecule, create_minimal_basis, compute_overlap_integrals, 
    gaussian_3d, analyze_overlaps
)

def compute_overlap_matrix(basis_functions: List[Dict], molecule) -> torch.Tensor:
    """
    Compute the overlap matrix W_μν = ∫ω_μ(r)ω_ν(r)dr using numerical integration.
    
    Args:
        basis_functions: list of basis function definitions
        molecule: molecular data for probe coordinates and volume element
    
    Returns:
        overlap_matrix: W_μν matrix (N_basis, N_basis)
    """
    n_basis = len(basis_functions)
    print(f"Computing {n_basis}x{n_basis} overlap matrix...")
    
    # Get probe coordinates for integration
    probe_coords = molecule.probe_coords  # (N_probes, 3)
    
    # Estimate volume element
    if hasattr(molecule, 'grid_size'):
        grid_size = molecule.grid_size[0]
        cell_volume = torch.det(molecule.cell[0]).abs()
        volume_element = cell_volume / torch.prod(grid_size.float())
    else:
        cell_volume = torch.det(molecule.cell[0]).abs()
        n_probes = len(probe_coords)
        volume_element = cell_volume / n_probes
    
    # Initialize overlap matrix
    overlap_matrix = torch.zeros(n_basis, n_basis)
    
    # Compute all basis function values at probe points first (for efficiency)
    print("  Evaluating all basis functions at probe points...")
    basis_values = torch.zeros(n_basis, len(probe_coords))
    
    for i, basis_func in enumerate(basis_functions):
        basis_values[i] = gaussian_3d(
            probe_coords,
            basis_func['center'],
            basis_func['exponent'],
            basis_func['l'],
            basis_func['m'],
            basis_func['n']
        )
        if i % 10 == 0:
            print(f"    Evaluated {i+1}/{n_basis} basis functions")
    
    # Compute overlap matrix: W_μν = ∫ω_μ(r)ω_ν(r)dr ≈ Σ ω_μ(r_i) * ω_ν(r_i) * ΔV
    print("  Computing overlap matrix elements...")
    for mu in range(n_basis):
        for nu in range(n_basis):
            overlap_matrix[mu, nu] = torch.sum(basis_values[mu] * basis_values[nu]) * volume_element
        
        if mu % 5 == 0:
            print(f"    Completed row {mu+1}/{n_basis}")
    
    return overlap_matrix, basis_values

def solve_coefficients(overlap_matrix: torch.Tensor, overlap_integrals: torch.Tensor, 
                      regularization: float = 1e-8) -> torch.Tensor:
    """
    Solve the linear system W_μν * c_ν = O_μ for coefficients c_ν.
    
    Args:
        overlap_matrix: W_μν matrix (N_basis, N_basis)
        overlap_integrals: O_μ vector (N_basis,)
        regularization: regularization parameter for numerical stability
    
    Returns:
        coefficients: c_ν vector (N_basis,)
    """
    print("Solving linear system for coefficients...")
    
    # Add regularization to diagonal for numerical stability
    W_reg = overlap_matrix + regularization * torch.eye(overlap_matrix.shape[0])
    
    print(f"  Overlap matrix condition number: {torch.linalg.cond(overlap_matrix):.2e}")
    print(f"  Regularized matrix condition number: {torch.linalg.cond(W_reg):.2e}")
    
    # Solve W * c = O
    try:
        coefficients = torch.linalg.solve(W_reg, overlap_integrals)
        print("  Successfully solved using direct method")
    except:
        print("  Direct solve failed, using least squares...")
        coefficients = torch.linalg.lstsq(W_reg, overlap_integrals)[0]
    
    return coefficients

def reconstruct_charge_density(coefficients: torch.Tensor, basis_values: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct charge density using ρ(r) = Σ c_ν * ω_ν(r).
    
    Args:
        coefficients: c_ν vector (N_basis,)
        basis_values: ω_ν(r) matrix (N_basis, N_probes)
    
    Returns:
        reconstructed_density: ρ(r) vector (N_probes,)
    """
    print("Reconstructing charge density...")
    
    # ρ(r) = Σ c_ν * ω_ν(r)
    reconstructed_density = torch.sum(coefficients.unsqueeze(1) * basis_values, dim=0)
    
    return reconstructed_density

def analyze_reconstruction(true_density: torch.Tensor, reconstructed_density: torch.Tensor, 
                         coefficients: torch.Tensor, labels: List[str], molecule):
    """
    Analyze the quality of charge density reconstruction.
    
    Args:
        true_density: true charge density values
        reconstructed_density: reconstructed charge density values
        coefficients: fitted coefficients
        labels: basis function labels
        molecule: molecular data for volume element calculation
    """
    print("\n" + "="*60)
    print("RECONSTRUCTION ANALYSIS")
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
    
    # Compute L1 error and normalize by number of electrons
    # L1 error should also account for volume element
    l1_error = np.sum(np.abs(recon_np - true_np)) 
    
    # Proper electron counting: ∫ρ(r)dr ≈ Σ ρ(r_i) * ΔV
    n_electrons_true = np.sum(true_np) * volume_element_np
    n_electrons_recon = np.sum(recon_np) * volume_element_np
    l1_error_per_electron = l1_error / n_electrons_true if n_electrons_true > 0 else float('inf')
    mae_per_electron = mae / n_electrons_true if n_electrons_true > 0 else float('inf')
    
    # Relative errors
    mean_true = np.mean(np.abs(true_np))
    relative_mae = mae / mean_true
    relative_rmse = rmse / mean_true
    relative_mae_per_electron = mae_per_electron / mean_true if mean_true > 0 else float('inf')
    
    print(f"Reconstruction Quality Metrics:")
    print(f"  R² score: {r2:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  MAE per electron: {mae_per_electron:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  L1 error: {l1_error:.6f}")
    print(f"  L1 error per electron: {l1_error_per_electron:.6f}")
    print(f"  Relative MAE: {relative_mae:.2%}")
    print(f"  Relative MAE per electron: {relative_mae_per_electron:.2%}")
    print(f"  Relative RMSE: {relative_rmse:.2%}")
    
    print(f"\nElectron counting (with volume element):")
    print(f"  Volume element: {volume_element_np:.6f} Ų")
    print(f"  Total electrons (true): {n_electrons_true:.4f}")
    print(f"  Total electrons (reconstructed): {n_electrons_recon:.4f}")
    print(f"  Electron difference: {n_electrons_recon - n_electrons_true:.4f}")
    print(f"  Electron conservation error: {100 * (n_electrons_recon - n_electrons_true) / n_electrons_true:.2%}")
    
    print(f"\nTrue density statistics:")
    print(f"  Min: {true_density.min():.6f}")
    print(f"  Max: {true_density.max():.6f}")
    print(f"  Mean: {true_density.mean():.6f}")
    print(f"  Std: {true_density.std():.6f}")
    print(f"  Sum (density): {true_density.sum():.6f}")
    print(f"  Integral (electrons): {n_electrons_true:.6f}")
    
    print(f"\nReconstructed density statistics:")
    print(f"  Min: {reconstructed_density.min():.6f}")
    print(f"  Max: {reconstructed_density.max():.6f}")
    print(f"  Mean: {reconstructed_density.mean():.6f}")
    print(f"  Std: {reconstructed_density.std():.6f}")
    print(f"  Sum (density): {reconstructed_density.sum():.6f}")
    print(f"  Integral (electrons): {n_electrons_recon:.6f}")
    
    # Analyze coefficients
    print(f"\nCoefficient Analysis:")
    print(f"  Number of coefficients: {len(coefficients)}")
    print(f"  Mean coefficient: {coefficients.mean():.6f}")
    print(f"  Std coefficient: {coefficients.std():.6f}")
    print(f"  Max coefficient: {coefficients.max():.6f} ({labels[coefficients.argmax()]})")
    print(f"  Min coefficient: {coefficients.min():.6f} ({labels[coefficients.argmin()]})")
    
    # Show largest coefficients
    abs_coeffs = coefficients.abs()
    sorted_indices = torch.argsort(abs_coeffs, descending=True)
    
    print(f"\nTop 10 largest coefficients (by absolute value):")
    print("-" * 50)
    for i in range(min(10, len(coefficients))):
        idx = sorted_indices[i]
        print(f"{labels[idx]:>15}: {coefficients[idx]:>12.6f} (|{abs_coeffs[idx]:.6f}|)")
    
    return r2, mae, rmse, relative_mae, relative_rmse, l1_error_per_electron, mae_per_electron, relative_mae_per_electron

def plot_3d_molecular_differences(true_density: torch.Tensor, reconstructed_density: torch.Tensor, 
                                molecule, max_points: int = 5000, difference_threshold: float = None):
    """
    Create a 3D plot showing the molecule with charge density differences color-coded.
    
    Args:
        true_density: true charge density values
        reconstructed_density: reconstructed charge density values
        molecule: molecular data with atomic positions and probe coordinates
        max_points: maximum number of probe points to plot
        difference_threshold: threshold for highlighting large differences (if None, uses std)
    """
    print("\nCreating 3D molecular difference plot...")
    
    # Calculate differences
    differences = (reconstructed_density - true_density).cpu().numpy()
    true_np = true_density.cpu().numpy()
    
    # Get atomic and probe coordinates
    real_atoms_mask = molecule.atom_types != 0
    atom_coords = molecule.coords[real_atoms_mask].cpu().numpy()
    atom_types = molecule.atom_types[real_atoms_mask].cpu().numpy()
    probe_coords = molecule.probe_coords.cpu().numpy();
    
    # Sample points if too many
    n_probes = len(probe_coords)
    if n_probes > max_points:
        indices = np.random.choice(n_probes, max_points, replace=False)
        probe_coords = probe_coords[indices]
        differences = differences[indices]
        true_np = true_np[indices]
    
    # Set difference threshold for highlighting
    if difference_threshold is None:
        difference_threshold = np.std(differences)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 6))
    
    # Plot 1: Atoms + All probe points colored by difference
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Plot atoms
    atom_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    unique_types = np.unique(atom_types)
    
    for i, atom_type in enumerate(unique_types):
        mask = atom_types == atom_type
        coords = atom_coords[mask]
        if len(coords) > 0:
            ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                       c=atom_colors[i % len(atom_colors)], s=200, alpha=0.8,
                       label=f'Atom Z={atom_type}', edgecolors='black', linewidth=1)
    
    # Plot probe points colored by difference
    scatter = ax1.scatter(probe_coords[:, 0], probe_coords[:, 1], probe_coords[:, 2],
                         c=differences, cmap='RdBu_r', s=20, alpha=0.6, vmin=-difference_threshold, 
                         vmax=difference_threshold)
    
    ax1.set_xlabel('X (Å)')
    ax1.set_ylabel('Y (Å)')
    ax1.set_zlabel('Z (Å)')
    ax1.set_title('All Differences\n(Red=Overestimated, Blue=Underestimated)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.5, aspect=20)
    cbar1.set_label('Density Difference')
    
    # Plot 2: Only large positive differences
    ax2 = fig.add_subplot(132, projection='3d')
    
    # Plot atoms again
    for i, atom_type in enumerate(unique_types):
        mask = atom_types == atom_type
        coords = atom_coords[mask]
        if len(coords) > 0:
            ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                       c=atom_colors[i % len(atom_colors)], s=200, alpha=0.8,
                       edgecolors='black', linewidth=1)
    
    # Plot only large positive differences (overestimated)
    large_positive = differences > difference_threshold
    if np.any(large_positive):
        ax2.scatter(probe_coords[large_positive, 0], probe_coords[large_positive, 1], 
                   probe_coords[large_positive, 2],
                   c=differences[large_positive], cmap='Reds', s=30, alpha=0.8)
    
    ax2.set_xlabel('X (Å)')
    ax2.set_ylabel('Y (Å)')
    ax2.set_zlabel('Z (Å)')
    ax2.set_title(f'Large Overestimations\n(>{difference_threshold:.4f})')
    
    # Plot 3: Only large negative differences
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Plot atoms again
    for i, atom_type in enumerate(unique_types):
        mask = atom_types == atom_type
        coords = atom_coords[mask]
        if len(coords) > 0:
            ax3.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                       c=atom_colors[i % len(atom_colors)], s=200, alpha=0.8,
                       edgecolors='black', linewidth=1)
    
    # Plot only large negative differences (underestimated)
    large_negative = differences < -difference_threshold
    if np.any(large_negative):
        ax3.scatter(probe_coords[large_negative, 0], probe_coords[large_negative, 1], 
                   probe_coords[large_negative, 2],
                   c=np.abs(differences[large_negative]), cmap='Blues', s=30, alpha=0.8)
    
    ax3.set_xlabel('X (Å)')
    ax3.set_ylabel('Y (Å)')
    ax3.set_zlabel('Z (Å)')
    ax3.set_title(f'Large Underestimations\n(<{-difference_threshold:.4f})')
    
    plt.tight_layout()
    plt.savefig('3d_charge_density_differences.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics about differences
    print(f"\n3D Visualization Statistics:")
    print(f"  Total probe points plotted: {len(differences)}")
    print(f"  Difference threshold: ±{difference_threshold:.4f}")
    print(f"  Large overestimations: {np.sum(large_positive)} ({100*np.sum(large_positive)/len(differences):.1f}%)")
    print(f"  Large underestimations: {np.sum(large_negative)} ({100*np.sum(large_negative)/len(differences):.1f}%)")
    print(f"  Max overestimation: {np.max(differences):.4f}")
    print(f"  Max underestimation: {np.min(differences):.4f}")
    
    print(f"3D plot saved as '3d_charge_density_differences.png'")

def plot_reconstruction_comparison(true_density: torch.Tensor, reconstructed_density: torch.Tensor, 
                                 molecule, max_points: int = 10000):
    """
    Create plots comparing true vs reconstructed charge density.
    
    Args:
        true_density: true charge density values
        reconstructed_density: reconstructed charge density values
        molecule: molecular data
        max_points: maximum number of points to plot
    """
    print("\nCreating comparison plots...")
    
    # Sample points for plotting if dataset is too large
    n_points = len(true_density)
    if n_points > max_points:
        indices = torch.randperm(n_points)[:max_points]
        true_plot = true_density[indices].cpu().numpy()
        recon_plot = reconstructed_density[indices].cpu().numpy()
    else:
        true_plot = true_density.cpu().numpy()
        recon_plot = reconstructed_density.cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Scatter plot: true vs reconstructed
    axes[0, 0].scatter(true_plot, recon_plot, alpha=0.5, s=1)
    axes[0, 0].plot([true_plot.min(), true_plot.max()], [true_plot.min(), true_plot.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True Charge Density')
    axes[0, 0].set_ylabel('Reconstructed Charge Density')
    axes[0, 0].set_title('True vs Reconstructed')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = recon_plot - true_plot
    axes[0, 1].scatter(true_plot, residuals, alpha=0.5, s=1)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('True Charge Density')
    axes[0, 1].set_ylabel('Residuals (Recon - True)')
    axes[0, 1].set_title('Residuals vs True')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histogram of true density
    axes[1, 0].hist(true_plot, bins=50, alpha=0.7, label='True', density=True)
    axes[1, 0].hist(recon_plot, bins=50, alpha=0.7, label='Reconstructed', density=True)
    axes[1, 0].set_xlabel('Charge Density')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Density Distributions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1, 1].hist(residuals, bins=50, alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Residual Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('charge_density_reconstruction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved as 'charge_density_reconstruction.png'")
    
    # Also create the 3D molecular plot
    plot_3d_molecular_differences(true_density, reconstructed_density, molecule)

def main(molecule_idx: int = 0, regularization: float = 1e-8, create_plots: bool = True):
    """
    Main function to perform charge density reconstruction.
    
    Args:
        molecule_idx: index of molecule to analyze
        regularization: regularization parameter for solving linear system
        create_plots: whether to create comparison plots
    """
    print("="*60)
    print("CHARGE DENSITY RECONSTRUCTION")
    print("="*60)
    
    # Load molecule and compute overlaps (reusing code from overlap.py)
    print("Step 1: Loading molecule and computing overlaps...")
    molecule = load_single_molecule(idx=molecule_idx)
    
    # Filter out virtual nodes
    real_atoms_mask = molecule.atom_types != 0
    real_atom_types = molecule.atom_types[real_atoms_mask]
    real_atom_coords = molecule.coords[real_atoms_mask]
    
    # Create basis functions
    basis_functions = create_minimal_basis(real_atom_types, real_atom_coords)
    print(f"Created {len(basis_functions)} basis functions")
    
    # Compute overlap integrals O_μ = ∫ρ(r)ω_μ(r)dr
    overlap_integrals, labels = compute_overlap_integrals(molecule, basis_functions)
    
    print(f"\nStep 2: Computing overlap matrix W_μν = ∫ω_μ(r)ω_ν(r)dr...")
    overlap_matrix, basis_values = compute_overlap_matrix(basis_functions, molecule)
    
    print(f"\nStep 3: Solving for coefficients c_ν...")
    coefficients = solve_coefficients(overlap_matrix, overlap_integrals, regularization)
    
    print(f"\nStep 4: Reconstructing charge density...")
    reconstructed_density = reconstruct_charge_density(coefficients, basis_values)
    
    print(f"\nStep 5: Analyzing reconstruction quality...")
    metrics = analyze_reconstruction(
        molecule.chg_labels, reconstructed_density, coefficients, labels, molecule
    )
    
    if create_plots:
        print(f"\nStep 6: Creating comparison plots...")
        plot_reconstruction_comparison(
            molecule.chg_labels, reconstructed_density, molecule
        )
    
    # Return results for further analysis
    results = {
        'molecule': molecule,
        'basis_functions': basis_functions,
        'overlap_integrals': overlap_integrals,
        'overlap_matrix': overlap_matrix,
        'coefficients': coefficients,
        'reconstructed_density': reconstructed_density,
        'true_density': molecule.chg_labels,
        'labels': labels,
        'metrics': {
            'r2': metrics[0],
            'mae': metrics[1], 
            'rmse': metrics[2],
            'relative_mae': metrics[3],
            'relative_rmse': metrics[4],
            'l1_error_per_electron': metrics[5],
            'mae_per_electron': metrics[6],
            'relative_mae_per_electron': metrics[7]
        }
    }
    
    print(f"\nCharge density reconstruction completed!")
    print(f"R² = {metrics[0]:.4f}, Relative RMSE = {metrics[4]:.2%}, L1 error/electron = {metrics[5]:.6f}, MAE/electron = {metrics[6]:.6f}, Rel MAE/electron = {metrics[7]:.2%}")
    
    return results

if __name__ == "__main__":
    # Run reconstruction with different regularization values
    regularizations = [1e-10, 1e-8, 1e-6, 1e-4]
    
    print("Testing different regularization parameters...")
    best_r2 = -np.inf
    best_results = None
    
    for reg in regularizations:
        print(f"\n{'='*60}")
        print(f"TESTING REGULARIZATION = {reg}")
        print(f"{'='*60}")
        
        try:
            results = main(molecule_idx=8, regularization=reg, create_plots=False)
            current_r2 = results['metrics']['r2']
            
            if current_r2 > best_r2:
                best_r2 = current_r2
                best_results = results
                
        except Exception as e:
            print(f"Failed with regularization {reg}: {e}")
            continue
    
    # Create plots for best results
    if best_results is not None:
        print(f"\n{'='*60}")
        print(f"BEST RESULTS (R² = {best_r2:.4f})")
        print(f"L1 error per electron = {best_results['metrics']['l1_error_per_electron']:.6f}")
        print(f"MAE per electron = {best_results['metrics']['mae_per_electron']:.6f}")
        print(f"Relative MAE per electron = {best_results['metrics']['relative_mae_per_electron']:.2%}")
        print(f"{'='*60}")
        
        plot_reconstruction_comparison(
            best_results['true_density'], 
            best_results['reconstructed_density'], 
            best_results['molecule']
        )
    
    print(f"\nFinal results stored in 'best_results' variable")

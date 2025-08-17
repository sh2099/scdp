import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from overlap_2 import (
    load_single_molecule, create_scdp_basis_functions, compute_overlap_integrals_gto, 
    analyze_overlaps_gto
)
from scdp.model.utils import get_nmape

# Set default dtype to double precision
torch.set_default_dtype(torch.float64)

def compute_overlap_matrix_gto(gto_dict: Dict, basis_info: List[Dict], molecule, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the overlap matrix W_μν = ∫ω_μ(r)ω_ν(r)dr using GTO basis functions.
    
    Args:
        gto_dict: dictionary of GTO objects by atomic number
        basis_info: list of basis function information
        molecule: molecular data for probe coordinates and volume element
        device: device to use for computation
    
    Returns:
        overlap_matrix: W_μν matrix (N_basis, N_basis)
        basis_values: evaluated basis function values (N_basis, N_probes)
    """
    n_basis = len(basis_info)
    print(f"Computing {n_basis}x{n_basis} overlap matrix using GTO basis functions...")
    print(f"Using device: {device}, precision: double (float64)")
    
    # Get probe coordinates for integration with double precision
    probe_coords = molecule.probe_coords.to(device).double()
    
    # Estimate volume element
    if hasattr(molecule, 'grid_size'):
        grid_size = molecule.grid_size[0]
        cell_volume = torch.det(molecule.cell[0]).abs().double()
        volume_element = cell_volume / torch.prod(grid_size.double())
    else:
        cell_volume = torch.det(molecule.cell[0]).abs().double()
        n_probes = len(probe_coords)
        volume_element = cell_volume / n_probes
    
    # Initialize overlap matrix with double precision
    overlap_matrix = torch.zeros(n_basis, n_basis, device=device, dtype=torch.float64)
    
    # Compute all basis function values at probe points first (for efficiency)
    print("  Evaluating all GTO basis functions at probe points...")
    basis_values = torch.zeros(n_basis, len(probe_coords), device=device, dtype=torch.float64)
    
    # Group basis functions by atomic number for efficient computation
    basis_by_element = {}
    for i, basis_func in enumerate(basis_info):
        z = basis_func['atomic_number']
        if z not in basis_by_element:
            basis_by_element[z] = []
        basis_by_element[z].append((i, basis_func))
    
    # Evaluate basis functions by element
    for z, basis_funcs_with_idx in basis_by_element.items():
        gto = gto_dict[z].to(device).double()  # Move to device with double precision
        
        for idx, basis_func in basis_funcs_with_idx:
            atom_coord = basis_func['center'].unsqueeze(0).to(device).double()  # (1, 3)
            orb_idx = basis_func['orbital_idx']
            
            # Use GTO forward method to compute basis function values
            n_probes_tensor = torch.tensor([len(probe_coords)], device=device)
            n_atoms_tensor = torch.tensor([1], device=device)
            
            # Get basis function values for this atom at all probe points
            gto_values = gto.forward(
                probe_coords=probe_coords,
                atom_coords=atom_coord,
                n_probes=n_probes_tensor,
                n_atoms=n_atoms_tensor,
                coeffs=None,
                expo_scaling=None,
                reorder=False,
                pbc=False,
                cell=None
            )  # Shape: (N_probes, gto.outdim)
            
            # Extract the specific orbital and ensure double precision
            basis_values[idx] = gto_values[:, orb_idx].double()
            
            if idx % 20 == 0:
                print(f"    Evaluated {idx+1}/{n_basis} basis functions")
    
    # Compute overlap matrix: W_μν = ∫ω_μ(r)ω_ν(r)dr ≈ Σ ω_μ(r_i) * ω_ν(r_i) * ΔV
    print("  Computing overlap matrix elements...")
    for mu in range(n_basis):
        for nu in range(n_basis):
            overlap_matrix[mu, nu] = torch.sum(basis_values[mu] * basis_values[nu]) * volume_element.double()
        
        if mu % 10 == 0:
            print(f"    Completed row {mu+1}/{n_basis}")
    
    return overlap_matrix.cpu(), basis_values.cpu()

def solve_coefficients(overlap_matrix: torch.Tensor, overlap_integrals: torch.Tensor, 
                      regularization: float = 1e-8, device: str = 'cuda') -> torch.Tensor:
    """
    Solve the linear system W_μν * c_ν = O_μ for coefficients c_ν.
    
    Args:
        overlap_matrix: W_μν matrix (N_basis, N_basis)
        overlap_integrals: O_μ vector (N_basis,)
        regularization: regularization parameter for numerical stability
        device: device to use for computation
    
    Returns:
        coefficients: c_ν vector (N_basis,)
    """
    print("Solving linear system for coefficients...")
    print(f"Using device: {device}, precision: double (float64)")
    
    # Move to device with double precision
    overlap_matrix = overlap_matrix.to(device).double()
    overlap_integrals = overlap_integrals.to(device).double()
    
    # Add regularization to diagonal for numerical stability
    W_reg = overlap_matrix + torch.tensor(regularization, dtype=torch.float64, device=device) * torch.eye(overlap_matrix.shape[0], device=device, dtype=torch.float64)
    
    print(f"  Overlap matrix condition number: {torch.linalg.cond(overlap_matrix):.2e}")
    print(f"  Regularized matrix condition number: {torch.linalg.cond(W_reg):.2e}")
    
    # Solve W * c = O
    try:
        coefficients = torch.linalg.solve(W_reg, overlap_integrals)
        print("  Successfully solved using direct method")
    except:
        print("  Direct solve failed, using least squares...")
        coefficients = torch.linalg.lstsq(W_reg, overlap_integrals)[0]
    
    return coefficients.cpu()

def reconstruct_charge_density(coefficients: torch.Tensor, basis_values: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
    """
    Reconstruct charge density using ρ(r) = Σ c_ν * ω_ν(r).
    
    Args:
        coefficients: c_ν vector (N_basis,)
        basis_values: ω_ν(r) matrix (N_basis, N_probes)
        device: device to use for computation
    
    Returns:
        reconstructed_density: ρ(r) vector (N_probes,)
    """
    print("Reconstructing charge density...")
    print(f"Using device: {device}, precision: double (float64)")
    
    # Move to device with double precision
    coefficients = coefficients.to(device).double()
    basis_values = basis_values.to(device).double()
    
    # ρ(r) = Σ c_ν * ω_ν(r)
    reconstructed_density = torch.sum(coefficients.unsqueeze(1) * basis_values, dim=0)
    
    return reconstructed_density.cpu()

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
    print("GTO RECONSTRUCTION ANALYSIS")
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
    
    # Fix: Use tensors for get_nmape function
    nmape = get_nmape(reconstructed_density, true_density).item()

    # Proper electron counting: ∫ρ(r)dr ≈ Σ ρ(r_i) * ΔV
    n_electrons_true = np.sum(true_np) * volume_element_np
    n_electrons_recon = np.sum(recon_np) * volume_element_np
    mae_per_electron = mae / n_electrons_true if n_electrons_true > 0 else float('inf')
    nmape_per_electron = nmape / n_electrons_true if n_electrons_true > 0 else float('inf')
    
    # Relative errors
    mean_true = np.mean(np.abs(true_np))
    relative_mae = mae / mean_true
    relative_rmse = rmse / mean_true
    
    print(f"Reconstruction Quality Metrics:")
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
        print(f"{labels[idx]:>20}: {coefficients[idx]:>12.6f} (|{abs_coeffs[idx]:.6f}|)")
    
    return r2, mae, rmse, relative_mae, relative_rmse, mae_per_electron, nmape

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
    ax1.set_title('All Differences (GTO Basis)\n(Red=Overestimated, Blue=Underestimated)')
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
    plt.savefig('3d_charge_density_differences_gto.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics about differences
    print(f"\n3D Visualization Statistics:")
    print(f"  Total probe points plotted: {len(differences)}")
    print(f"  Difference threshold: ±{difference_threshold:.4f}")
    print(f"  Large overestimations: {np.sum(large_positive)} ({100*np.sum(large_positive)/len(differences):.1f}%)")
    print(f"  Large underestimations: {np.sum(large_negative)} ({100*np.sum(large_negative)/len(differences):.1f}%)")
    print(f"  Max overestimation: {np.max(differences):.4f}")
    print(f"  Max underestimation: {np.min(differences):.4f}")
    
    print(f"3D plot saved as '3d_charge_density_differences_gto.png'")

def plot_reconstruction_comparison(true_density: torch.Tensor, reconstructed_density: torch.Tensor, 
                                 molecule, max_points: int = 10000):
    """
    Create plots comparing true vs reconstructed charge density.
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
    axes[0, 0].set_title('True vs Reconstructed (GTO Basis)')
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
    plt.savefig('charge_density_reconstruction_gto.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved as 'charge_density_reconstruction_gto.png'")
    
    # Also create the 3D molecular plot
    plot_3d_molecular_differences(true_density, reconstructed_density, molecule)

def main(molecule_idx: int = 0, regularization: float = 1e-8, create_plots: bool = True,
         basis_set_name: str = 'def2-svp', use_augmentation: bool = True, beta: float = 2.0,
         device: str = 'cuda'):
    """
    Main function to perform charge density reconstruction using GTO basis functions.
    
    Args:
        molecule_idx: index of molecule to analyze
        regularization: regularization parameter for solving linear system
        create_plots: whether to create comparison plots
        basis_set_name: name of basis set to use
        use_augmentation: whether to use even-tempered augmentation
        beta: parameter for even-tempered augmentation
        device: device to use for computation ('cuda' or 'cpu')
    """
    print("="*60)
    print("GTO CHARGE DENSITY RECONSTRUCTION")
    print("="*60)
    print(f"Using device: {device}, precision: double (float64)")
    
    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Load molecule
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
    
    # Compute overlap integrals O_μ = ∫ρ(r)ω_μ(r)dr
    overlap_integrals, labels = compute_overlap_integrals_gto(molecule, gto_dict, basis_info)
    
    print(f"\nStep 2: Computing overlap matrix W_μν = ∫ω_μ(r)ω_ν(r)dr...")
    overlap_matrix, basis_values = compute_overlap_matrix_gto(gto_dict, basis_info, molecule, device)
    
    print(f"\nStep 3: Solving for coefficients c_ν...")
    coefficients = solve_coefficients(overlap_matrix, overlap_integrals, regularization, device)
    
    print(f"\nStep 4: Reconstructing charge density...")
    reconstructed_density = reconstruct_charge_density(coefficients, basis_values, device)
    
    print(f"\nStep 5: Analyzing reconstruction quality...")
    metrics = analyze_reconstruction(
        molecule.chg_labels.double(), reconstructed_density.double(), coefficients.double(), labels, molecule
    )
    
    if create_plots:
        print(f"\nStep 6: Creating comparison plots...")
        plot_reconstruction_comparison(
            molecule.chg_labels.double(), reconstructed_density.double(), molecule
        )
    
    # Return results for further analysis
    results = {
        'molecule': molecule,
        'gto_dict': gto_dict,
        'basis_info': basis_info,
        'overlap_integrals': overlap_integrals.double(),
        'overlap_matrix': overlap_matrix.double(),
        'coefficients': coefficients.double(),
        'reconstructed_density': reconstructed_density.double(),
        'true_density': molecule.chg_labels.double(),
        'labels': labels,
        'basis_set_name': basis_set_name,
        'use_augmentation': use_augmentation,
        'beta': beta,
        'device': device,
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
    
    print(f"\nGTO charge density reconstruction completed!")
    print(f"Basis set: {basis_set_name}, Augmentation: {use_augmentation}")
    print(f"Device: {device}, Precision: double (float64)")
    print(f"R² = {metrics[0]:.4f}, Relative RMSE = {metrics[4]:.2%}, MAE/electron = {metrics[5]:.6f}")
    
    return results

def plot_regularization_analysis(reg_results: List[Dict]):
    """
    Plot NMAPE vs regularization values to analyze regularization effects.
    
    Args:
        reg_results: List of results dictionaries from different regularization values
    """
    print("\nCreating regularization analysis plot...")
    
    # Extract data for plotting
    regularizations = [result['regularization'] for result in reg_results]
    nmapes = [result['metrics']['nmape'] for result in reg_results]
    r2_scores = [result['metrics']['r2'] for result in reg_results]
    mae_per_electrons = [result['metrics']['mae_per_electron'] for result in reg_results]
    
    # Filter NMAPE values to show only those <= 1
    nmapes_filtered = [min(nmape, 1.0) for nmape in nmapes]
    nmape_clipped = [nmape > 1.0 for nmape in nmapes]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: NMAPE vs Regularization (capped at 1.0)
    colors = ['red' if clipped else 'blue' for clipped in nmape_clipped]
    scatter = axes[0, 0].scatter(regularizations, nmapes_filtered, c=colors, s=60, alpha=0.7)
    for i, (x, y, clipped) in enumerate(zip(regularizations, nmapes_filtered, nmape_clipped)):
        if clipped:
            axes[0, 0].annotate(f'{nmapes[i]:.2f}', (x, y), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8, color='red')
    
    axes[0, 0].plot(regularizations, nmapes_filtered, 'k-', alpha=0.3, linewidth=1)
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel('Regularization Parameter')
    axes[0, 0].set_ylabel('NMAPE (capped at 1.0)')
    axes[0, 0].set_title('NMAPE vs Regularization')
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add legend for color coding
    red_patch = plt.matplotlib.patches.Patch(color='red', label='NMAPE > 1.0 (clipped)')
    blue_patch = plt.matplotlib.patches.Patch(color='blue', label='NMAPE ≤ 1.0')
    axes[0, 0].legend(handles=[blue_patch, red_patch], fontsize=8)
    
    # Plot 2: R² vs Regularization
    axes[0, 1].semilogx(regularizations, r2_scores, 'o-', color='green', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Regularization Parameter')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].set_title('R² Score vs Regularization')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: MAE per electron vs Regularization
    axes[1, 0].semilogx(regularizations, mae_per_electrons, 'o-', color='red', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Regularization Parameter')
    axes[1, 0].set_ylabel('MAE per Electron')
    axes[1, 0].set_title('MAE per Electron vs Regularization')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Combined normalized metrics (with NMAPE capped)
    # Normalize all metrics to [0, 1] for comparison
    nmapes_norm = np.array(nmapes_filtered)
    nmapes_norm = (nmapes_norm - nmapes_norm.min()) / (nmapes_norm.max() - nmapes_norm.min()) if nmapes_norm.max() > nmapes_norm.min() else nmapes_norm
    
    r2_norm = np.array(r2_scores)
    r2_norm = 1 - (r2_norm - r2_norm.min()) / (r2_norm.max() - r2_norm.min()) if r2_norm.max() > r2_norm.min() else r2_norm  # Invert so lower is better
    
    mae_norm = np.array(mae_per_electrons)
    mae_norm = (mae_norm - mae_norm.min()) / (mae_norm.max() - mae_norm.min()) if mae_norm.max() > mae_norm.min() else mae_norm
    
    axes[1, 1].semilogx(regularizations, nmapes_norm, 'o-', label='NMAPE (norm, capped)', linewidth=2, markersize=6)
    axes[1, 1].semilogx(regularizations, r2_norm, 's-', label='1-R² (norm)', linewidth=2, markersize=6)
    axes[1, 1].semilogx(regularizations, mae_norm, '^-', label='MAE/e⁻ (norm)', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Regularization Parameter')
    axes[1, 1].set_ylabel('Normalized Metric (0=best)')
    axes[1, 1].set_title('Normalized Metrics vs Regularization')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regularization_analysis_gto.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print(f"\nRegularization Analysis Summary:")
    print("-" * 90)
    print(f"{'Regularization':>12} {'NMAPE':>8} {'Capped?':>8} {'R²':>8} {'MAE/e⁻':>10} {'Best?':>6}")
    print("-" * 90)
    
    best_nmape_idx = np.argmin(nmapes_filtered)
    best_r2_idx = np.argmax(r2_scores)
    
    for i, (reg, nmape, nmape_filt, r2, mae_e, clipped) in enumerate(zip(regularizations, nmapes, nmapes_filtered, r2_scores, mae_per_electrons, nmape_clipped)):
        best_marker = ""
        if i == best_nmape_idx:
            best_marker += "N"
        if i == best_r2_idx:
            best_marker += "R"
        
        clipped_marker = "Yes" if clipped else "No"
        print(f"{reg:>12.0e} {nmape:>8.4f} {clipped_marker:>8} {r2:>8.4f} {mae_e:>10.6f} {best_marker:>6}")
    
    print("-" * 90)
    print("Best markers: N=Best NMAPE (capped), R=Best R²")
    print("NMAPE values > 1.0 are shown in red and capped at 1.0 in plots")
    print(f"Plot saved as 'regularization_analysis_gto.png'")

def test_multiple_regularizations(molecule_idx: int = 0, 
                                basis_set_name: str = 'def2-svp', 
                                use_augmentation: bool = True, 
                                beta: float = 2.0,
                                regularizations: List[float] = None,
                                device: str = 'cuda'):
    """
    Test multiple regularization values and analyze their effects.
    
    Args:
        molecule_idx: index of molecule to analyze
        basis_set_name: name of basis set to use
        use_augmentation: whether to use even-tempered augmentation
        beta: parameter for even-tempered augmentation
        regularizations: list of regularization values to test
    
    Returns:
        List of results dictionaries for each regularization value
    """
    if regularizations is None:
        regularizations = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
    
    print("="*80)
    print("TESTING MULTIPLE REGULARIZATION VALUES")
    print("="*80)
    print(f"Testing regularizations: {regularizations}")
    print(f"Basis set: {basis_set_name}, Augmentation: {use_augmentation}")
    print(f"Using device: {device}, precision: double (float64)")
    
    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Load molecule and create basis functions once
    print("\nStep 1: Loading molecule and creating GTO basis functions...")
    molecule = load_single_molecule(idx=molecule_idx)
    
    real_atoms_mask = molecule.atom_types != 0
    real_atom_types = molecule.atom_types[real_atoms_mask]
    real_atom_coords = molecule.coords[real_atoms_mask]
    
    gto_dict, basis_info = create_scdp_basis_functions(
        real_atom_types, real_atom_coords,
        basis_set_name=basis_set_name,
        use_augmentation=use_augmentation,
        beta=beta
    )
    print(f"Created {len(basis_info)} GTO basis functions")
    
    # Compute overlap integrals and matrix once
    print("\nStep 2: Computing overlap integrals and matrix...")
    overlap_integrals, labels = compute_overlap_integrals_gto(molecule, gto_dict, basis_info)
    overlap_matrix, basis_values = compute_overlap_matrix_gto(gto_dict, basis_info, molecule, device)
    
    # Test each regularization value
    reg_results = []
    
    for i, reg in enumerate(regularizations):
        print(f"\n{'='*60}")
        print(f"TESTING REGULARIZATION {i+1}/{len(regularizations)}: {reg:.0e}")
        print(f"{'='*60}")
        
        try:
            # Solve for coefficients with current regularization
            coefficients = solve_coefficients(overlap_matrix, overlap_integrals, reg, device)
            
            # Reconstruct charge density
            reconstructed_density = reconstruct_charge_density(coefficients, basis_values, device)
            
            # Analyze reconstruction quality
            metrics = analyze_reconstruction(
                molecule.chg_labels.double(), reconstructed_density.double(), coefficients.double(), labels, molecule
            )
            
            # Store results
            result = {
                'regularization': reg,
                'coefficients': coefficients.double(),
                'reconstructed_density': reconstructed_density.double(),
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
            reg_results.append(result)
            
            print(f"Summary: R² = {metrics[0]:.4f}, NMAPE = {metrics[6]:.4f}, MAE/e⁻ = {metrics[5]:.6f}")
            
        except Exception as e:
            print(f"Failed with regularization {reg}: {e}")
            continue
    
    if len(reg_results) > 1:
        # Create regularization analysis plot
        plot_regularization_analysis(reg_results)
        
        # Find best regularization
        nmapes = [r['metrics']['nmape'] for r in reg_results]
        best_idx = np.argmin(nmapes)
        best_reg = reg_results[best_idx]
        
        print(f"\n{'='*80}")
        print(f"BEST REGULARIZATION ANALYSIS")
        print(f"{'='*80}")
        print(f"Best regularization (by NMAPE): {best_reg['regularization']:.0e}")
        print(f"Best NMAPE: {best_reg['metrics']['nmape']:.4f}")
        print(f"Best R²: {best_reg['metrics']['r2']:.4f}")
        print(f"Best MAE/electron: {best_reg['metrics']['mae_per_electron']:.6f}")
        
        # Store additional information
        for result in reg_results:
            result.update({
                'molecule': molecule,
                'gto_dict': gto_dict,
                'basis_info': basis_info,
                'overlap_integrals': overlap_integrals,
                'overlap_matrix': overlap_matrix,
                'true_density': molecule.chg_labels,
                'labels': labels,
                'basis_set_name': basis_set_name,
                'use_augmentation': use_augmentation,
                'beta': beta
            })
        
        return reg_results, best_reg
    else:
        print("Not enough successful results to perform analysis")
        return reg_results, None

if __name__ == "__main__":
    # Choose analysis mode
    analysis_mode = "regularization"  # Options: "regularization", "basis_sets", "both"
    mol_idx = 8  # Example molecule index
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*80)
    print("CUDA + DOUBLE PRECISION RECONSTRUCTION ANALYSIS")
    print("="*80)
    print(f"Using device: {device}, precision: double (float64)")
    print("Note: NMAPE plots are capped at 1.0 for better visualization")

    if analysis_mode in ["regularization", "both"]:
        # Test regularization analysis
        print("="*80)
        print("REGULARIZATION ANALYSIS MODE")
        print("="*80)
        
        # Test multiple regularization values
        regularizations = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
        
        reg_results, best_result = test_multiple_regularizations(
            molecule_idx=mol_idx,
            basis_set_name="def2-svp",
            use_augmentation=True,
            beta=2.0,
            regularizations=regularizations,
            device=device
        )
        
        if best_result is not None:
            print(f"\n{'='*80}")
            print(f"CREATING PLOTS FOR BEST REGULARIZATION")
            print(f"{'='*80}")
            
            # Create detailed plots for best regularization
            plot_reconstruction_comparison(
                best_result['true_density'], 
                best_result['reconstructed_density'], 
                best_result['molecule']
            )
            
            print(f"\nRegularization analysis completed!")
            print(f"Results stored in 'reg_results' and 'best_result' variables")
        else:
            print(f"\nRegularization analysis failed - no successful results")
    
    if analysis_mode in ["basis_sets", "both"]:
        # Test different GTO basis set configurations
        print(f"\n{'='*80}")
        print("BASIS SET COMPARISON MODE")
        print("="*80)
        
        if analysis_mode == "both":
            reg = best_result['regularization'] if 'best_result' in locals() else 1e-8
        else:
            reg = 1e-8
        test_configs = [
            {"basis_set_name": "def2-svp", "use_augmentation": True, "beta": 2.0, "regularization": reg, "device": device},
            {"basis_set_name": "def2-QZVPPD", "use_augmentation": True, "beta": 2.0, "regularization": reg, "device": device},
        ]
        
        print("Testing different GTO basis set configurations...")
        best_r2 = -np.inf
        best_basis_results = None
        basis_comparison_results = []
        
        for i, config in enumerate(test_configs):
            print(f"\n{'='*60}")
            print(f"TESTING BASIS CONFIGURATION {i+1}: {config}")
            print(f"{'='*60}")
            
            try:
                results = main(molecule_idx=mol_idx, create_plots=False, **config)
                current_r2 = results['metrics']['r2']
                
                # Store for comparison
                basis_comparison_results.append({
                    'config': config,
                    'results': results,
                    'r2': current_r2,
                    'nmape': results['metrics']['nmape'],
                    'mae_per_electron': results['metrics']['mae_per_electron']
                })
                
                if current_r2 > best_r2:
                    best_r2 = current_r2
                    best_basis_results = results
                    
            except Exception as e:
                print(f"Configuration failed: {e}")
                continue
        
        # Create basis set comparison plot
        if len(basis_comparison_results) > 1:
            print(f"\n{'='*60}")
            print("CREATING BASIS SET COMPARISON PLOT")
            print(f"{'='*60}")
            
            # Create comparison plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            config_names = []
            r2_scores = []
            nmapes = []
            mae_per_electrons = []
            
            for result in basis_comparison_results:
                config = result['config']
                config_name = f"{config['basis_set_name']}"
                if config['use_augmentation']:
                    config_name += f"+Aug(β={config['beta']})"
                config_names.append(config_name)
                r2_scores.append(result['r2'])
                nmapes.append(result['nmape'])
                mae_per_electrons.append(result['mae_per_electron'])
            
            # Plot 1: R² comparison
            bars1 = axes[0].bar(range(len(config_names)), r2_scores, alpha=0.7, color='skyblue')
            axes[0].set_xlabel('Basis Set Configuration')
            axes[0].set_ylabel('R² Score')
            axes[0].set_title('R² Score Comparison')
            axes[0].set_xticks(range(len(config_names)))
            axes[0].set_xticklabels(config_names, rotation=45, ha='right')
            axes[0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars1, r2_scores):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{value:.4f}', ha='center', va='bottom')
            
            # Plot 2: NMAPE comparison (capped at 1.0)
            nmapes_capped = [min(nmape, 1.0) for nmape in nmapes]
            colors = ['red' if nmape > 1.0 else 'lightcoral' for nmape in nmapes]
            bars2 = axes[1].bar(range(len(config_names)), nmapes_capped, alpha=0.7, color=colors)
            axes[1].set_xlabel('Basis Set Configuration')
            axes[1].set_ylabel('NMAPE (capped at 1.0)')
            axes[1].set_title('NMAPE Comparison')
            axes[1].set_xticks(range(len(config_names)))
            axes[1].set_xticklabels(config_names, rotation=45, ha='right')
            axes[1].set_ylim(0, 1.0)
            axes[1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value, original_value in zip(bars2, nmapes_capped, nmapes):
                height = bar.get_height()
                if original_value > 1.0:
                    label = f'{original_value:.2f}'
                    color = 'red'
                else:
                    label = f'{value:.4f}'
                    color = 'black'
                axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           label, ha='center', va='bottom', color=color, fontweight='bold' if original_value > 1.0 else 'normal')
            
            # Plot 3: MAE per electron comparison
            bars3 = axes[2].bar(range(len(config_names)), mae_per_electrons, alpha=0.7, color='lightgreen')
            axes[2].set_xlabel('Basis Set Configuration')
            axes[2].set_ylabel('MAE per Electron')
            axes[2].set_title('MAE per Electron Comparison')
            axes[2].set_xticks(range(len(config_names)))
            axes[2].set_xticklabels(config_names, rotation=45, ha='right')
            axes[2].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars3, mae_per_electrons):
                height = bar.get_height()
                axes[2].text(bar.get_x() + bar.get_width()/2., height + max(mae_per_electrons)*0.01,
                           f'{value:.6f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('basis_set_comparison_gto.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Basis set comparison plot saved as 'basis_set_comparison_gto.png'")
            print("Note: NMAPE values > 1.0 are shown in red and capped at 1.0")
            
            # Print summary table
            print(f"\nBasis Set Comparison Summary:")
            print("-" * 90)
            print(f"{'Configuration':>25} {'R²':>8} {'NMAPE':>8} {'Capped?':>8} {'MAE/e⁻':>10}")
            print("-" * 90)
            
            for result, name, nmape in zip(basis_comparison_results, config_names, nmapes):
                clipped = "Yes" if nmape > 1.0 else "No"
                print(f"{name:>25} {result['r2']:>8.4f} {nmape:>8.4f} {clipped:>8} {result['mae_per_electron']:>10.6f}")
            
            print("-" * 90)
            print("NMAPE values > 1.0 are highlighted and capped at 1.0 in plots")
        
        # Create plots for best basis set results
        if best_basis_results is not None:
            print(f"\n{'='*60}")
            print(f"BEST BASIS SET RESULTS (R² = {best_r2:.4f})")
            best_config = next(r['config'] for r in basis_comparison_results if r['r2'] == best_r2)
            print(f"Best configuration: {best_config}")
            print(f"MAE per electron = {best_basis_results['metrics']['mae_per_electron']:.6f}")
            print(f"NMAPE = {best_basis_results['metrics']['nmape']:.4f}")
            print(f"{'='*60}")
            
            plot_reconstruction_comparison(
                best_basis_results['true_density'], 
                best_basis_results['reconstructed_density'], 
                best_basis_results['molecule']
            )
            
            print(f"\nBasis set comparison completed!")
            print(f"Results stored in 'basis_comparison_results' and 'best_basis_results' variables")
        else:
            print(f"\nBasis set comparison failed - no successful results")
    
    print(f"\nAnalysis completed! Available variables:")
    if analysis_mode in ["regularization", "both"]:
        print(f"  - reg_results: regularization analysis results")
        print(f"  - best_result: best regularization result")
    if analysis_mode in ["basis_sets", "both"]:
        print(f"  - basis_comparison_results: basis set comparison results")
        print(f"  - best_basis_results: best basis set result")
import pandas as pd
import json
import numpy as np
from pathlib import Path
import glob
from typing import Dict, List, Tuple

def find_multi_gpu_csv_files(directory: str = "/export/home/hmichael/scdp/mol_nmape") -> List[str]:
    """
    Find all CSV files starting with 'multi_gpu_molecule_results'.
    
    Args:
        directory: Directory to search in
    
    Returns:
        List of CSV file paths
    """
    pattern = f"{directory}/multi_gpu_molecule_results_*.csv"
    csv_files = glob.glob(pattern)
    csv_files.sort()  # Sort for consistent processing order
    
    print(f"Found {len(csv_files)} multi_gpu_molecule_results CSV files:")
    for file in csv_files:
        print(f"  {file}")
    
    return csv_files

def combine_csv_files(csv_files: List[str]) -> pd.DataFrame:
    """
    Combine multiple CSV files, ensuring no molecule appears more than once.
    
    Args:
        csv_files: List of CSV file paths
    
    Returns:
        Combined DataFrame with unique molecules
    """
    all_data = []
    processed_molecules = set()
    duplicate_count = 0
    
    print("\nCombining CSV files...")
    
    for file_path in csv_files:
        print(f"Processing {file_path}...")
        
        try:
            df = pd.read_csv(file_path)
            print(f"  Loaded {len(df)} rows")
            
            # Check for duplicates within this file
            file_molecules = df['molecule_idx'].unique()
            file_duplicates = len(df) - len(file_molecules) * len(df['regularization'].unique())
            if file_duplicates > 0:
                print(f"  Warning: {file_duplicates} duplicate molecule-regularization pairs in this file")
            
            # Filter out molecules we've already seen
            before_filter = len(df)
            df_filtered = df[~df['molecule_idx'].isin(processed_molecules)]
            after_filter = len(df_filtered)
            
            removed = before_filter - after_filter
            if removed > 0:
                duplicate_count += removed
                print(f"  Removed {removed} rows for molecules already processed")
            
            # Add new molecules to our tracking set
            new_molecules = set(df_filtered['molecule_idx'].unique())
            processed_molecules.update(new_molecules)
            print(f"  Added {len(new_molecules)} new molecules")
            
            all_data.append(df_filtered)
            
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid CSV files were processed")
    
    # Combine all DataFrames
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\nCombination summary:")
    print(f"  Total rows after combination: {len(combined_df)}")
    print(f"  Unique molecules: {len(combined_df['molecule_idx'].unique())}")
    print(f"  Unique regularizations: {len(combined_df['regularization'].unique())}")
    print(f"  Duplicate rows removed: {duplicate_count}")
    
    # Final check for duplicates
    duplicates = combined_df.duplicated(subset=['molecule_idx', 'regularization'])
    if duplicates.any():
        print(f"  Warning: {duplicates.sum()} duplicate molecule-regularization pairs remain!")
        # Remove duplicates, keeping first occurrence
        combined_df = combined_df.drop_duplicates(subset=['molecule_idx', 'regularization'], keep='first')
        print(f"  After removing duplicates: {len(combined_df)} rows")
    
    return combined_df

def create_summary_statistics(df: pd.DataFrame) -> Dict:
    """
    Create summary statistics from the combined DataFrame.
    
    Args:
        df: Combined DataFrame

    Returns:
        Dictionary with summary statistics
    """
    print("\nCreating summary statistics...")

    # Get unique molecules and regularizations
    molecule_indices = sorted(df['molecule_idx'].unique())
    regularizations = sorted(df['regularization'].unique())

    print(f"Molecules: {len(molecule_indices)}")
    print(f"Regularizations: {len(regularizations)}")

    # Create matrices for NMAPE and R²
    nmape_matrix = np.full((len(molecule_indices), len(regularizations)), np.nan)
    r2_matrix = np.full((len(molecule_indices), len(regularizations)), np.nan)

    # Fill matrices
    for _, row in df.iterrows():
        mol_idx = molecule_indices.index(row['molecule_idx'])
        reg_idx = regularizations.index(row['regularization'])
        
        # Handle failed results
        if not (np.isinf(row['nmape']) or np.isnan(row['nmape'])):
            nmape_matrix[mol_idx, reg_idx] = row['nmape']
        if not (np.isinf(row['r2']) or np.isnan(row['r2']) or row['r2'] < -1e10):
            r2_matrix[mol_idx, reg_idx] = row['r2']

    # Calculate averages
    molecule_averages = []
    for i in range(len(molecule_indices)):
        mol_nmapes = nmape_matrix[i, :]
        valid_nmapes = mol_nmapes[~np.isnan(mol_nmapes)]
        mol_avg = np.mean(valid_nmapes) if len(valid_nmapes) > 0 else float('inf')
        molecule_averages.append(float(mol_avg))  # Convert to Python float

    regularization_averages = []
    for j in range(len(regularizations)):
        reg_nmapes = nmape_matrix[:, j]
        valid_nmapes = reg_nmapes[~np.isnan(reg_nmapes)]
        reg_avg = np.mean(valid_nmapes) if len(valid_nmapes) > 0 else float('inf')
        regularization_averages.append(float(reg_avg))  # Convert to Python float

    # Overall average
    valid_mol_averages = [avg for avg in molecule_averages if not np.isinf(avg)]
    overall_avg = float(np.mean(valid_mol_averages)) if valid_mol_averages else float('inf')

    # Find best results
    best_mol_idx = np.argmin(molecule_averages)
    best_reg_idx = np.nanargmin(regularization_averages)

    # Find best single result
    best_single_nmape = float(np.nanmin(nmape_matrix))
    best_single_idx = np.unravel_index(np.nanargmin(nmape_matrix), nmape_matrix.shape)
    best_single_mol = int(molecule_indices[best_single_idx[0]])  # Convert to Python int
    best_single_reg = float(regularizations[best_single_idx[1]])  # Convert to Python float

    # Find worst single result
    worst_single_nmape = float(np.nanmax(nmape_matrix))
    worst_single_idx = np.unravel_index(np.nanargmax(nmape_matrix), nmape_matrix.shape)
    worst_single_mol = int(molecule_indices[worst_single_idx[0]])  # Convert to Python int
    worst_single_reg = float(regularizations[worst_single_idx[1]])  # Convert to Python float

    # Convert numpy arrays to lists and ensure all values are JSON serializable
    def convert_to_json_serializable(obj):
        """Convert numpy types to JSON serializable Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    summary = {
        'regularization_averages': regularization_averages,
        'overall_average': overall_avg,
        'best_molecule': int(molecule_indices[best_mol_idx]),  # Convert to Python int
        'best_regularization': float(regularizations[best_reg_idx]),  # Convert to Python float
        'best_single_nmape': best_single_nmape,
        'best_single_molecule': best_single_mol,
        'best_single_regularization': best_single_reg,
        'worst_single_nmape': worst_single_nmape,
        'worst_single_molecule': worst_single_mol,
        'worst_single_regularization': worst_single_reg,
        'molecule_indices': convert_to_json_serializable(molecule_indices),
        'regularizations': convert_to_json_serializable(regularizations),
        'total_molecules': len(molecule_indices),
        'total_regularizations': len(regularizations),
        'total_data_points': len(df),
        'nmape_matrix': convert_to_json_serializable(nmape_matrix),
        'r2_matrix': convert_to_json_serializable(r2_matrix),
        'molecule_averages': molecule_averages,
    }

    print(f"\nSummary statistics:")
    print(f"  Overall average NMAPE: {overall_avg:.4f}")
    print(f"  Best molecule: {molecule_indices[best_mol_idx]} (avg NMAPE: {molecule_averages[best_mol_idx]:.4f})")
    print(f"  Best regularization: {regularizations[best_reg_idx]:.0e} (avg NMAPE: {regularization_averages[best_reg_idx]:.4f})")
    print(f"  Best single result: {best_single_nmape:.4f} (Mol {best_single_mol}, reg {best_single_reg:.0e})")
    print(f"  Worst single result: {worst_single_nmape:.4f} (Mol {worst_single_mol}, reg {worst_single_reg:.0e})")

    return summary

def print_detailed_summary(df: pd.DataFrame, summary: Dict):
    """
    Print detailed summary table.
    
    Args:
        df: Combined DataFrame
        summary: Summary statistics dictionary
    """
    print(f"\n{'='*100}")
    print("DETAILED SUMMARY TABLE")
    print(f"{'='*100}")
    
    molecule_indices = summary['molecule_indices']
    regularizations = summary['regularizations']
    nmape_matrix = np.array(summary['nmape_matrix'])
    
    # Print header
    header = f"{'Molecule':>8}"
    for reg in regularizations:
        header += f"{reg:>10.0e}"
    header += f"{'Average':>10} {'N_basis':>8}"
    print(header)
    print("-" * len(header))
    
    # Print data for each molecule
    for i, mol_idx in enumerate(molecule_indices):
        row = f"{mol_idx:>8}"
        
        # Get basis count for this molecule
        mol_data = df[df['molecule_idx'] == mol_idx]
        n_basis = mol_data['n_basis'].iloc[0] if len(mol_data) > 0 else 0
        
        # NMAPE values for each regularization
        mol_nmapes = []
        for j, reg in enumerate(regularizations):
            nmape_val = nmape_matrix[i, j]
            if not np.isnan(nmape_val) and not np.isinf(nmape_val):
                row += f"{nmape_val:>10.4f}"
                mol_nmapes.append(nmape_val)
            else:
                row += f"{'FAIL':>10}"
        
        # Average
        mol_avg = np.mean(mol_nmapes) if mol_nmapes else float('inf')
        if not np.isinf(mol_avg):
            row += f"{mol_avg:>10.4f}"
        else:
            row += f"{'FAIL':>10}"
        
        row += f"{n_basis:>8}"
        print(row)
    
    # Print regularization averages
    print("-" * len(header))
    row = f"{'Average':>8}"
    for reg_avg in summary['regularization_averages']:
        if not np.isinf(reg_avg):
            row += f"{reg_avg:>10.4f}"
        else:
            row += f"{'FAIL':>10}"
    
    row += f"{summary['overall_average']:>10.4f}"
    row += f"{'-':>8}"
    print(row)
    print("-" * len(header))

def main():
    """Main function to combine results and create summary."""
    print("="*80)
    print("COMBINING MULTI-GPU MOLECULE RESULTS")
    print("="*80)
    
    # Find all CSV files
    csv_files = find_multi_gpu_csv_files()
    
    if not csv_files:
        print("No multi_gpu_molecule_results CSV files found!")
        return
    
    # Combine CSV files
    combined_df = combine_csv_files(csv_files)
    
    # Save combined CSV
    output_csv = "/export/home/hmichael/scdp/combined_multi_gpu_results.csv"
    combined_df.to_csv(output_csv, index=False)
    print(f"\nSaved combined results to: {output_csv}")
    
    # Create summary statistics
    summary = create_summary_statistics(combined_df)
    
    # Save summary JSON
    output_json = "/export/home/hmichael/scdp/combined_multi_gpu_results_summary.json"
    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary statistics to: {output_json}")
    
    # Print detailed summary
    print_detailed_summary(combined_df, summary)
    
    # Additional analysis
    print(f"\n{'='*80}")
    print("ADDITIONAL ANALYSIS")
    print(f"{'='*80}")
    
    # Distribution of basis sizes
    basis_sizes = combined_df['n_basis'].value_counts().sort_index()
    print(f"\nBasis size distribution:")
    for basis_size, count in basis_sizes.head(10).items():
        print(f"  {basis_size:>4} basis functions: {count:>3} molecules")
    if len(basis_sizes) > 10:
        print(f"  ... and {len(basis_sizes) - 10} more sizes")
    
    print(f"\nBasis size statistics:")
    print(f"  Min: {combined_df['n_basis'].min()}")
    print(f"  Max: {combined_df['n_basis'].max()}")
    print(f"  Mean: {combined_df['n_basis'].mean():.1f}")
    print(f"  Median: {combined_df['n_basis'].median():.1f}")
    
    # NMAPE distribution
    valid_nmapes = combined_df[~combined_df['nmape'].isin([np.inf, -np.inf, np.nan])]['nmape']
    if len(valid_nmapes) > 0:
        print(f"\nNMAPE statistics:")
        print(f"  Count: {len(valid_nmapes)}")
        print(f"  Min: {valid_nmapes.min():.4f}")
        print(f"  Max: {valid_nmapes.max():.4f}")
        print(f"  Mean: {valid_nmapes.mean():.4f}")
        print(f"  Median: {valid_nmapes.median():.4f}")
        print(f"  Std: {valid_nmapes.std():.4f}")
        
        # Count by quality ranges
        excellent = (valid_nmapes <= 0.2).sum()
        good = ((valid_nmapes > 0.2) & (valid_nmapes <= 0.3)).sum()
        fair = ((valid_nmapes > 0.3) & (valid_nmapes <= 0.5)).sum()
        poor = (valid_nmapes > 0.5).sum()
        
        print(f"\nNMAPE quality distribution:")
        print(f"  Excellent (≤0.2): {excellent:>3} ({100*excellent/len(valid_nmapes):>5.1f}%)")
        print(f"  Good (0.2-0.3):   {good:>3} ({100*good/len(valid_nmapes):>5.1f}%)")
        print(f"  Fair (0.3-0.5):   {fair:>3} ({100*fair/len(valid_nmapes):>5.1f}%)")
        print(f"  Poor (>0.5):      {poor:>3} ({100*poor/len(valid_nmapes):>5.1f}%)")
    
    print(f"\nCombination completed successfully!")
    print(f"Combined data available in: {output_csv}")
    print(f"Summary statistics available in: {output_json}")
    
    return combined_df, summary

if __name__ == "__main__":
    combined_df, summary = main()

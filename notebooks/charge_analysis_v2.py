import torch
import numpy as np
import time
import csv
from pathlib import Path
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
    device: str = 'cpu',
    devices: Optional[List[str]] = None,
    exclude_gpus: Optional[List[int]] = None
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
        devices: List of device strings to use (if None, auto-detect)
        exclude_gpus: List of GPU indices to exclude from auto-detection
        
    Returns:
        Dictionary with analysis results
    """
    # Start total timing
    total_start_time = time.time()
    
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
    
    # Determine devices for parallel compute
    if devices is None:
        if torch.cuda.is_available():
            all_gpus = list(range(torch.cuda.device_count()))
            if exclude_gpus is not None:
                available_gpus = [i for i in all_gpus if i not in exclude_gpus]
                if not available_gpus:
                    print("Warning: All GPUs excluded, falling back to CPU")
                    devices = ["cpu"]
                else:
                    devices = [f"cuda:{i}" for i in available_gpus]
                    if exclude_gpus:
                        print(f"Excluded GPUs: {exclude_gpus}")
            else:
                devices = [f"cuda:{i}" for i in all_gpus]
        else:
            devices = ["cpu"]
    print(f"Detected devices for processing: {devices}")

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
    
    # Step 4: Compute overlap integrals with probe blocking and multi-GPU
    print("\n4. Computing overlap integrals...")
    start_time = time.time()
    overlap_integrals = compute_overlap_integrals_scdp(
        molecule, gto_dict, atom_coords, atom_types, max_probes_per_chunk, 
        devices=devices, exclude_gpus=exclude_gpus
    )
    overlap_time = time.time() - start_time
    print(f"Overlap computation time: {overlap_time:.2f}s")
 
    # Step 5: Compute overlap matrix with probe blocking and multi-GPU
    print("\n5. Computing overlap matrix...")
    start_time = time.time()
    overlap_matrix = compute_overlap_matrix_scdp(
        gto_dict, atom_coords, atom_types, molecule, max_probes_per_chunk, 
        devices=devices, exclude_gpus=exclude_gpus
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
            
            # Reconstruct charge density (multi-GPU)
            print("    Reconstructing charge density...")
            start_time = time.time()
            reconstructed = reconstruct_density_scdp(
                coefficients.cpu(), gto_dict, atom_coords.cpu(), atom_types.cpu(),
                molecule.cpu(), max_probes_per_chunk, devices=devices, exclude_gpus=exclude_gpus
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
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Molecule: {molecule_idx}")
    print(f"Basis functions: {total_basis_funcs}")
    print(f"Probe points: {len(molecule.chg_labels)}")
    print(f"Probe chunks used: {len(probe_chunks) if 'probe_chunks' in locals() else 'N/A'}")
    print(f"Center info: {center_info}")
    print(f"Total molecule computation time: {total_time:.2f}s")
    
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
            'matrix_time': matrix_time,
            'total_time': total_time
        },
        'probe_blocking': {
            'max_probes_per_chunk': max_probes_per_chunk,
            'total_probes': len(molecule.chg_labels)
        }
    }

def read_molecule_indices_from_csv(csv_file: str, molecule_column: str = 'molecule_idx') -> List[int]:
    """
    Read molecule indices from a CSV file.
    
    Args:
        csv_file: Path to CSV file
        molecule_column: Name of column containing molecule indices
        
    Returns:
        List of unique molecule indices
    """
    molecule_indices = []
    csv_path = Path(csv_file)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        if molecule_column not in reader.fieldnames:
            raise ValueError(f"Column '{molecule_column}' not found in CSV. Available columns: {reader.fieldnames}")
        
        for row in reader:
            try:
                mol_idx = int(row[molecule_column])
                if mol_idx not in molecule_indices:
                    molecule_indices.append(mol_idx)
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not parse molecule index from row {row}: {e}")
                continue
    
    molecule_indices.sort()
    print(f"Read {len(molecule_indices)} unique molecule indices from {csv_file}")
    print(f"Range: {min(molecule_indices)} to {max(molecule_indices)}")
    return molecule_indices

def save_results_to_csv(all_results: Dict, output_file: str, base_config: Dict, append_mode: bool = False):
    """
    Save analysis results to CSV file with comprehensive metadata.
    
    Args:
        all_results: Dictionary of results keyed by molecule_idx
        output_file: Output CSV file path
        base_config: Base configuration used for analysis
        append_mode: If True, append to existing file; if False, create new file
    """
    output_path = Path(output_file)
    
    fieldnames = [
        'molecule_idx', 'nmape', 'r2', 'mae', 'mse', 'regularization',
        'n_basis_functions', 'n_probes', 'total_time_s', 'overlap_time_s', 'matrix_time_s',
        'basis_set_name', 'use_augmentation', 'beta', 'use_vnodes', 'override_atom_type',
        'max_probes_per_chunk', 'n_devices_used', 'success'
    ]
    
    # Determine file mode and whether to write header
    mode = 'a' if append_mode else 'w'
    write_header = not append_mode or not output_path.exists()
    
    with open(output_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()
        
        for mol_idx, result in all_results.items():
            # Find best result (lowest NMAPE)
            successful_results = {k: v for k, v in result['results'].items() if v['success']}
            
            if successful_results:
                best_reg = min(successful_results.keys(), key=lambda x: successful_results[x]['nmape'])
                best_result = successful_results[best_reg]
                success = True
                nmape = best_result['nmape']
                r2 = best_result['r2']
                mae = best_result['mae']
                mse = best_result['mse']
            else:
                success = False
                best_reg = list(result['results'].keys())[0] if result['results'] else 'N/A'
                nmape = float('inf')
                r2 = -float('inf')
                mae = float('inf')
                mse = float('inf')
            
            # Determine number of devices used
            n_devices_used = len(base_config.get('devices', [])) if 'devices' in base_config else 'auto'
            
            row = {
                'molecule_idx': mol_idx,
                'nmape': nmape,
                'r2': r2,
                'mae': mae,
                'mse': mse,
                'regularization': best_reg,
                'n_basis_functions': result['basis_info']['total_functions'],
                'n_probes': result['probe_blocking']['total_probes'],
                'total_time_s': result['timings']['total_time'],
                'overlap_time_s': result['timings']['overlap_time'],
                'matrix_time_s': result['timings']['matrix_time'],
                'basis_set_name': result['basis_info']['basis_set_name'],
                'use_augmentation': result['basis_info']['use_augmentation'],
                'beta': result['basis_info']['beta'],
                'use_vnodes': base_config.get('use_vnodes', False),
                'override_atom_type': base_config.get('override_atom_type', None),
                'max_probes_per_chunk': result['probe_blocking']['max_probes_per_chunk'],
                'n_devices_used': n_devices_used,
                'success': success
            }
            
            writer.writerow(row)
    
    print(f"Results {'appended to' if append_mode else 'saved to'}: {output_path}")

def clear_memory():
    """Clear GPU and system memory."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

def main(input_csv: str = None, molecule_column: str = 'molecule_idx'):
    """Main function with CSV input support and incremental result saving."""
    
    # Base configuration - applied to all molecules
    base_config = {
        'regularizations': [1e-10],
        'basis_set_name': 'def2-QZVPP',
        'use_augmentation': True,
        'use_vnodes': True,
        'override_atom_type': 8,
        'max_probes_per_chunk': 40000,
        'exclude_gpus': [4, 7, 6]  
    }
    
    # Determine molecule indices source
    if input_csv:
        print(f"Reading molecule indices from CSV: {input_csv}")
        molecule_indices = read_molecule_indices_from_csv(input_csv, molecule_column)
        # Limit for testing - remove this line for full processing
        molecule_indices = molecule_indices[50:] if len(molecule_indices) > 20 else molecule_indices
    else:
        print("Using default molecule indices")
        molecule_indices = [34075, 5, 343, 11797, 39941, 17, 1000, 2000, 3000]
    
    print(f"\n{'#'*100}")
    print(f"RUNNING MULTI-MOLECULE ANALYSIS WITH PROBE BLOCKING")
    print(f"{'#'*100}")
    print(f"Base configuration: {base_config}")
    print(f"Analyzing {len(molecule_indices)} molecules: {molecule_indices}")
    print(f"Saving results every 10 molecules to prevent data loss")
    print(f"{'#'*100}")
    
    # Generate output filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    vnode_suffix = "_vnodes" if base_config['use_vnodes'] else ""
    override_suffix = f"_override{base_config['override_atom_type']}" if base_config['override_atom_type'] else ""
    input_suffix = f"_from_{Path(input_csv).stem}" if input_csv else ""
    output_csv = f"charge_analysis_results{vnode_suffix}{override_suffix}{input_suffix}_{timestamp}.csv"
    
    # Store results for batches of molecules
    batch_results = {}
    all_successful_molecules = []
    all_failed_molecules = []
    
    total_analysis_start = time.time()
    save_interval = 5  # Save every 10 molecules
    
    for i, mol_idx in enumerate(molecule_indices):
        print(f"\n{'='*80}")
        print(f"MOLECULE {i+1}/{len(molecule_indices)}: Index {mol_idx}")
        print(f"{'='*80}")
        
        try:
            # Create config for this molecule
            config = base_config.copy()
            config['molecule_idx'] = mol_idx
            
            mol_start_time = time.time()
            result = analyze_molecule_charge_reconstruction(**config)
            mol_time = time.time() - mol_start_time
            
            # Store results in current batch
            batch_results[mol_idx] = result
            all_successful_molecules.append(mol_idx)
            
            # Print brief summary for this molecule
            successful = sum(1 for r in result['results'].values() if r['success'])
            total = len(result['results'])
            probe_info = result['probe_blocking']
            
            print(f"\nMolecule {mol_idx} completed in {mol_time:.1f}s: {successful}/{total} successful reconstructions")
            print(f"Processed {probe_info['total_probes']} probes, Total time: {result['timings']['total_time']:.2f}s")
            
            # Show best result if available
            if successful > 0:
                successful_results = {k: v for k, v in result['results'].items() if v['success']}
                best_reg = min(successful_results.keys(), key=lambda x: successful_results[x]['nmape'])
                best_result = successful_results[best_reg]
                print(f"Best result: λ={best_reg:.2e}, NMAPE={best_result['nmape']:.4f}, R²={best_result['r2']:.4f}")
            
        except Exception as e:
            print(f"Molecule {mol_idx} failed: {e}")
            all_failed_molecules.append(mol_idx)
            # Create dummy result for failed molecule
            batch_results[mol_idx] = {
                'results': {1e-10: {'success': False, 'nmape': float('inf'), 'r2': -float('inf'), 'mae': float('inf'), 'mse': float('inf')}},
                'basis_info': {'total_functions': 0, 'basis_set_name': base_config['basis_set_name'], 'use_augmentation': base_config['use_augmentation'], 'beta': base_config.get('beta', 2.0)},
                'timings': {'total_time': 0, 'overlap_time': 0, 'matrix_time': 0},
                'probe_blocking': {'total_probes': 0, 'max_probes_per_chunk': base_config['max_probes_per_chunk']}
            }
        
        # Save results every save_interval molecules or at the end
        if (i + 1) % save_interval == 0 or (i + 1) == len(molecule_indices):
            print(f"\n{'='*60}")
            print(f"SAVING BATCH RESULTS (molecules {max(0, i-save_interval+1)+1} to {i+1})")
            print(f"{'='*60}")
            
            # Save current batch to CSV
            append_mode = (i + 1) > save_interval  # Append if not the first batch
            save_results_to_csv(batch_results, output_csv, base_config, append_mode=append_mode)
            
            # Clear memory and reset batch
            print(f"Clearing memory and batch results...")
            del batch_results
            clear_memory()
            batch_results = {}
            
            print(f"Progress: {i+1}/{len(molecule_indices)} molecules processed")
            elapsed = time.time() - total_analysis_start
            if i + 1 < len(molecule_indices):
                estimated_total = elapsed * len(molecule_indices) / (i + 1)
                remaining = estimated_total - elapsed
                print(f"Elapsed: {elapsed:.1f}s, Estimated remaining: {remaining:.1f}s")
    
    total_analysis_time = time.time() - total_analysis_start
    
    # Final summary - read back from CSV for verification
    print(f"\n{'='*80}")
    print("READING FINAL RESULTS FROM SAVED CSV")
    print(f"{'='*80}")
    
    try:
        import pandas as pd
        df = pd.read_csv(output_csv)
        successful_from_csv = df[df['success'] == True]
        failed_from_csv = df[df['success'] == False]
        
        print(f"CSV verification:")
        print(f"  Total rows in CSV: {len(df)}")
        print(f"  Successful: {len(successful_from_csv)}")
        print(f"  Failed: {len(failed_from_csv)}")
        
        if len(successful_from_csv) > 0:
            print(f"\nAggregate Statistics from CSV:")
            print(f"NMAPE: {successful_from_csv['nmape'].mean():.4f} ± {successful_from_csv['nmape'].std():.4f}")
            print(f"R²: {successful_from_csv['r2'].mean():.4f} ± {successful_from_csv['r2'].std():.4f}")
            print(f"Computation time: {successful_from_csv['total_time_s'].mean():.1f} ± {successful_from_csv['total_time_s'].std():.1f}s")
            print(f"Basis functions: {successful_from_csv['n_basis_functions'].mean():.0f} ± {successful_from_csv['n_basis_functions'].std():.0f}")
            
            print(f"\nBest performer: Molecule {successful_from_csv.loc[successful_from_csv['nmape'].idxmin(), 'molecule_idx']} (NMAPE: {successful_from_csv['nmape'].min():.4f})")
            print(f"Worst performer: Molecule {successful_from_csv.loc[successful_from_csv['nmape'].idxmax(), 'molecule_idx']} (NMAPE: {successful_from_csv['nmape'].max():.4f})")
        
    except ImportError:
        print("pandas not available for CSV verification, but results are saved")
    except Exception as e:
        print(f"Error reading CSV for verification: {e}")
    
    # Summary of all results
    print(f"\n{'#'*100}")
    print(f"MULTI-MOLECULE ANALYSIS SUMMARY")
    print(f"{'#'*100}")
    print(f"Total analysis time: {total_analysis_time:.1f}s")
    print(f"Successful molecules: {len(all_successful_molecules)}/{len(molecule_indices)}")
    print(f"Failed molecules: {len(all_failed_molecules)}")
    
    if all_successful_molecules:
        print(f"\nSuccessful molecules: {all_successful_molecules}")
    
    if all_failed_molecules:
        print(f"\nFailed molecules: {all_failed_molecules}")
    
    print(f"\nResults saved to: {output_csv}")
    print(f"Results were saved incrementally every {save_interval} molecules")
    print(f"\n{'#'*100}")
    print("MULTI-MOLECULE ANALYSIS COMPLETED")
    print(f"{'#'*100}")
    
    return all_successful_molecules, all_failed_molecules, output_csv


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-molecule charge density reconstruction analysis")
    parser.add_argument("--input_csv", type=str, default=None,
                       help="Path to CSV file containing molecule indices")
    parser.add_argument("--molecule_column", type=str, default="molecule_idx",
                       help="Name of column containing molecule indices (default: molecule_idx)")
    
    args = parser.parse_args()
    
    # Run with CSV input if provided
    if args.input_csv:
        all_results, successful, failed, output_file = main(
            input_csv=args.input_csv,
            molecule_column=args.molecule_column
        )
    else:
        all_results, successful, failed, output_file = main()
    
    print(f"\nAnalysis completed!")
    print(f"Successful: {len(successful)}, Failed: {len(failed)}")
    print(f"Results saved to: {output_file}")

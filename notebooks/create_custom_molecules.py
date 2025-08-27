import torch
import numpy as np
import time
import csv
from pathlib import Path
from typing import List, Dict, Optional, Union

from notebooks.mol_loader_v2 import load_molecule_with_override, get_atom_centers_and_types
from notebooks.overlap_v2 import create_gto_basis, compute_overlap_integrals_scdp
from overlap_pred.custom_data import CustomMolecule

# Set default dtype to double precision
torch.set_default_dtype(torch.float64)

def process_molecule_to_custom_data(
    molecule_idx: int,
    basis_set_name: str = 'def2-QZVPPD',
    use_augmentation: bool = True,
    beta: float = 2.0,
    use_vnodes: bool = True,
    override_atom_type: Optional[int] = None,
    vnode_elem: int = 1,
    max_probes_per_chunk: int = 50000,
    device: str = 'cpu',
    devices: Optional[List[str]] = None,
    exclude_gpus: Optional[List[int]] = None
) -> CustomMolecule:
    """
    Process a single molecule to create a CustomMolecule with computed overlap integral.
    
    Args:
        molecule_idx: Index of molecule to process
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
        CustomMolecule object with computed overlap integral
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING MOLECULE {molecule_idx}")
    print(f"Basis set: {basis_set_name}")
    print(f"Augmentation: {use_augmentation} (β={beta})")
    print(f"Virtual nodes: {use_vnodes}")
    print(f"Atom type override: {override_atom_type}")
    print(f"{'='*60}")
    
    # Step 1: Load molecule with optional atom type override
    print("\n1. Loading molecule...")
    start_time = time.time()
    molecule = load_molecule_with_override(
        idx=molecule_idx, 
        vnode=use_vnodes, 
        override_atom_type=override_atom_type
    )
    load_time = time.time() - start_time
    print(f"   Load time: {load_time:.2f}s")
    
    # Step 2: Get atom centers and types for basis placement
    print("\n2. Extracting atom centers and types...")
    atom_coords, atom_types, center_info = get_atom_centers_and_types(
        molecule, 
        use_vnodes=use_vnodes, 
        override_atom_type=override_atom_type
    )
    print(f"   {center_info}")
    
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
                    print("   Warning: All GPUs excluded, falling back to CPU")
                    devices = ["cpu"]
                else:
                    devices = [f"cuda:{i}" for i in available_gpus]
                    if exclude_gpus:
                        print(f"   Excluded GPUs: {exclude_gpus}")
            else:
                devices = [f"cuda:{i}" for i in all_gpus]
        else:
            devices = ["cpu"]
    print(f"   Using devices: {devices}")

    # Step 3: Construct GTO basis
    print("\n3. Constructing GTO basis...")
    start_time = time.time()
    gto_dict = create_gto_basis(
        atom_types=atom_types,
        atom_coords=atom_coords,
        basis_set_name=basis_set_name,
        use_augmentation=use_augmentation,
        beta=beta,
        vnode_elem=vnode_elem
    )
    basis_time = time.time() - start_time
    
    # Calculate total basis functions
    total_basis_funcs = 0
    for t in torch.unique(atom_types):
        t_str = str(int(t.item()))
        if t_str in gto_dict:
            n_atoms_of_type = int((atom_types == t).sum().item())
            total_basis_funcs += n_atoms_of_type * gto_dict[t_str].outdim
    print(f"   Total basis functions: {total_basis_funcs}")
    print(f"   Basis construction time: {basis_time:.2f}s")
    
    # Step 4: Compute overlap integrals (this is the key computation we need)
    print("\n4. Computing overlap integrals...")
    start_time = time.time()
    overlap_integrals = compute_overlap_integrals_scdp(
        molecule, gto_dict, atom_coords, atom_types, max_probes_per_chunk, 
        devices=devices, exclude_gpus=exclude_gpus
    )
    overlap_time = time.time() - start_time
    print(f"   Overlap computation time: {overlap_time:.2f}s")
    print(f"   Overlap integrals shape: {overlap_integrals.shape}")
    
    # Compute a scalar overlap integral value (e.g., sum of all overlaps)
    overlap_integral_value = float(overlap_integrals.sum().item())
    print(f"   Total overlap integral: {overlap_integral_value:.6e}")
    
    # Step 5: Create CustomMolecule object
    print("\n5. Creating CustomMolecule object...")
    custom_molecule = CustomMolecule.from_scdp_data(
        molecule.cpu(),  # Move back to CPU for storage
        overlap_integral=overlap_integral_value,
        keep_probe_grid=False  # Don't keep heavy grid data
    )
    
    # Add additional metadata about the computation
    custom_molecule.metadata.update({
        'basis_set_name': basis_set_name,
        'use_augmentation': use_augmentation,
        'beta': beta,
        'total_basis_functions': total_basis_funcs,
        'processing_time': {
            'load_time': load_time,
            'basis_time': basis_time,
            'overlap_time': overlap_time
        },
        'devices_used': devices,
        'max_probes_per_chunk': max_probes_per_chunk,
        'center_info': center_info
    })
    
    print(f"   CustomMolecule created: {custom_molecule}")
    return custom_molecule

def process_molecules_batch(
    molecule_indices: List[int],
    output_dir: str,
    basis_set_name: str = 'def2-QZVPPD',
    use_augmentation: bool = True,
    beta: float = 2.0,
    use_vnodes: bool = True,
    override_atom_type: Optional[int] = None,
    vnode_elem: int = 1,
    max_probes_per_chunk: int = 50000,
    device: str = 'cpu',
    devices: Optional[List[str]] = None,
    exclude_gpus: Optional[List[int]] = None,
    save_interval: int = 10
) -> Dict[int, str]:
    """
    Process a batch of molecules and save them as CustomMolecule pickle files.
    
    Args:
        molecule_indices: List of molecule indices to process
        output_dir: Directory to save pickle files
        (other args same as process_molecule_to_custom_data)
        save_interval: How often to print progress updates
        
    Returns:
        Dictionary mapping molecule_idx to output file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*80}")
    print(f"BATCH PROCESSING {len(molecule_indices)} MOLECULES")
    print(f"Output directory: {output_path}")
    print(f"Basis set: {basis_set_name}")
    print(f"Virtual nodes: {use_vnodes}")
    print(f"Atom type override: {override_atom_type}")
    print(f"{'#'*80}")
    
    results = {}
    successful = 0
    failed = 0
    
    batch_start_time = time.time()
    
    for i, mol_idx in enumerate(molecule_indices):
        try:
            # Process molecule
            custom_mol = process_molecule_to_custom_data(
                molecule_idx=mol_idx,
                basis_set_name=basis_set_name,
                use_augmentation=use_augmentation,
                beta=beta,
                use_vnodes=use_vnodes,
                override_atom_type=override_atom_type,
                vnode_elem=vnode_elem,
                max_probes_per_chunk=max_probes_per_chunk,
                device=device,
                devices=devices,
                exclude_gpus=exclude_gpus
            )
            
            # Generate output filename
            vnode_suffix = "_vnodes" if use_vnodes else ""
            override_suffix = f"_override{override_atom_type}" if override_atom_type else ""
            aug_suffix = "_aug" if use_augmentation else ""
            filename = f"molecule_{mol_idx}{vnode_suffix}{override_suffix}{aug_suffix}_{basis_set_name}.pkl"
            output_file = output_path / filename
            
            # Save to pickle
            custom_mol.save_pickle(output_file)
            
            results[mol_idx] = str(output_file)
            successful += 1
            
            print(f"   ✓ Saved to: {output_file}")
            
        except Exception as e:
            print(f"   ✗ Failed to process molecule {mol_idx}: {e}")
            results[mol_idx] = f"FAILED: {e}"
            failed += 1
        
        # Progress update
        if (i + 1) % save_interval == 0 or (i + 1) == len(molecule_indices):
            elapsed = time.time() - batch_start_time
            progress = (i + 1) / len(molecule_indices)
            estimated_total = elapsed / progress if progress > 0 else 0
            remaining = estimated_total - elapsed
            
            print(f"\n{'='*40}")
            print(f"PROGRESS: {i+1}/{len(molecule_indices)} ({progress*100:.1f}%)")
            print(f"Successful: {successful}, Failed: {failed}")
            print(f"Elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s")
            print(f"{'='*40}")
        
        # Clear memory periodically
        if (i + 1) % 20 == 0:
            clear_memory()
    
    total_time = time.time() - batch_start_time
    
    print(f"\n{'#'*80}")
    print(f"BATCH PROCESSING COMPLETED")
    print(f"Total time: {total_time:.1f}s")
    print(f"Successful: {successful}/{len(molecule_indices)}")
    print(f"Failed: {failed}/{len(molecule_indices)}")
    print(f"Output directory: {output_path}")
    print(f"{'#'*80}")
    
    return results

def read_molecule_indices_from_csv(csv_file: str, molecule_column: str = 'molecule_idx') -> List[int]:
    """Read molecule indices from a CSV file."""
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
    return molecule_indices

def clear_memory():
    """Clear GPU and system memory."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

def save_processing_log(results: Dict[int, str], log_file: str):
    """Save processing results to a CSV log file."""
    log_path = Path(log_file)
    
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['molecule_idx', 'status', 'output_file'])
        
        for mol_idx, result in results.items():
            if result.startswith('FAILED:'):
                status = 'FAILED'
                output_file = result
            else:
                status = 'SUCCESS'
                output_file = result
            
            writer.writerow([mol_idx, status, output_file])
    
    print(f"Processing log saved to: {log_path}")

def main():
    """Main function to process molecules and create CustomMolecule pickle files."""
    
    # Configuration
    config = {
        'basis_set_name': 'def2-QZVPP',
        'use_augmentation': True,
        'beta': 2.0,
        'use_vnodes': True,
        'override_atom_type': 8,  # Oxygen
        'max_probes_per_chunk': 40000,
        'exclude_gpus': [4, 6, 7],
        'device': 'cpu'
    }
    
    # Test with a small set of molecules first
    test_molecules = [34075, 5, 343, 11797, 39941]
    
    # Create timestamped output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    vnode_suffix = "_vnodes" if config['use_vnodes'] else ""
    override_suffix = f"_override{config['override_atom_type']}" if config['override_atom_type'] else ""
    output_dir = f"custom_molecules{vnode_suffix}{override_suffix}_{config['basis_set_name']}_{timestamp}"
    
    print(f"Starting molecule processing...")
    print(f"Configuration: {config}")
    print(f"Test molecules: {test_molecules}")
    
    # Process molecules
    results = process_molecules_batch(
        molecule_indices=test_molecules,
        output_dir=output_dir,
        **config
    )
    
    # Save processing log
    log_file = f"{output_dir}/processing_log.csv"
    save_processing_log(results, log_file)
    
    # Test loading one of the saved files
    successful_files = [path for path in results.values() if not path.startswith('FAILED:')]
    if successful_files:
        test_file = successful_files[0]
        print(f"\n{'='*60}")
        print(f"TESTING PICKLE LOAD")
        print(f"Loading: {test_file}")
        
        try:
            loaded_mol = CustomMolecule.load_pickle(test_file)
            print(f"✓ Successfully loaded: {loaded_mol}")
            print(f"Overlap integral: {loaded_mol.overlap_integral}")
            print(f"Metadata keys: {list(loaded_mol.metadata.keys())}")
            print(f"Basis info: {loaded_mol.metadata.get('basis_set_name', 'N/A')}")
        except Exception as e:
            print(f"✗ Failed to load: {e}")
    
    print(f"\nProcessing completed!")
    print(f"Results saved to: {output_dir}")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create CustomMolecule pickle files with overlap integrals")
    parser.add_argument("--input_csv", type=str, default=None,
                       help="Path to CSV file containing molecule indices")
    parser.add_argument("--molecule_column", type=str, default="molecule_idx",
                       help="Name of column containing molecule indices")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for pickle files")
    parser.add_argument("--basis_set", type=str, default="def2-QZVPP",
                       help="Basis set name")
    parser.add_argument("--use_vnodes", action="store_true",
                       help="Use virtual nodes")
    parser.add_argument("--override_atom_type", type=int, default=None,
                       help="Override all atom types to this value")
    parser.add_argument("--max_chunk_size", type=int, default=50000,
                       help="Maximum probes per chunk")
    
    args = parser.parse_args()
    
    if args.input_csv:
        # Read molecules from CSV
        molecule_indices = read_molecule_indices_from_csv(args.input_csv, args.molecule_column)
        
        # Configuration from command line
        config = {
            'basis_set_name': args.basis_set,
            'use_augmentation': True,
            'beta': 2.0,
            'use_vnodes': args.use_vnodes,
            'override_atom_type': args.override_atom_type,
            'max_probes_per_chunk': args.max_chunk_size,
            'exclude_gpus': [4, 6, 7],
            'device': 'cpu'
        }
        
        # Output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            vnode_suffix = "_vnodes" if config['use_vnodes'] else ""
            override_suffix = f"_override{config['override_atom_type']}" if config['override_atom_type'] else ""
            input_suffix = f"_from_{Path(args.input_csv).stem}"
            output_dir = f"custom_molecules{vnode_suffix}{override_suffix}{input_suffix}_{timestamp}"
        
        print(f"Processing {len(molecule_indices)} molecules from CSV")
        print(f"Output directory: {output_dir}")
        
        # Process molecules
        results = process_molecules_batch(
            molecule_indices=molecule_indices,
            output_dir=output_dir,
            **config
        )
        
        # Save log
        log_file = f"{output_dir}/processing_log.csv"
        save_processing_log(results, log_file)
        
        print(f"Processing completed! Results in: {output_dir}")
    else:
        # Run with default test molecules
        results = main()

import torch
import numpy as np
import time
import csv
from pathlib import Path
from typing import List, Dict, Optional, Union

from overlap_pred.load_mol import load_molecule_with_override, get_atom_centers_and_types
from overlap_pred.comp_overlap import create_gto_basis, compute_overlap_integrals_2d_scdp
from overlap_pred.custom_gto_basis import create_gto_basis_custom, validate_custom_basis_parameters
# Use the new final CustomMolecule implementation (v2)
from overlap_pred.custom_data_2 import CustomMolecule

# Helper utilities to keep generator concise
from overlap_pred.gen_helpers import (
    build_mapping_storage,
    compute_nonpadded_stats,
    safe_shape_from_mapping,
    format_safe_molecule_id,
    build_custom_molecule_from_compressed,
)

# Set default dtype to double precision
torch.set_default_dtype(torch.float64)

from overlap_pred.overlap_analysis import print_overlap_analysis_table

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
    exclude_gpus: Optional[List[int]] = None,
    use_custom_gtos: bool = False,
    custom_gto_config: Optional[Dict] = None,
    show_L_values: Optional[List[int]] = None,
    silent_basis_analysis: bool = False,
    silent_gpu: bool = False,
    show_overlap_analysis: bool = True,
    use_direct_indexing: bool = False
) -> CustomMolecule:
    """
    Process a single molecule to create a CustomMolecule with computed overlap integral.
    
    Args:
        molecule_idx: Index of molecule to process
        silent_basis_analysis: If True, minimize basis analysis output
        silent_gpu: If True, minimize GPU computation output
        show_overlap_analysis: If True, show detailed overlap analysis table
        use_direct_indexing: If True, use direct dataset indexing (bypasses train/val/test splits)
        (other args same as before)
    
    Returns:
        CustomMolecule object with computed overlap integral
    """
    # Step 1: Load molecule with optional atom type override
    print("\n1. Loading molecule...")
    start_time = time.time()
    
    if use_direct_indexing:
        from overlap_pred.load_mol import load_molecule_direct
        molecule = load_molecule_direct(
            dataset_idx=molecule_idx,
            vnode=use_vnodes, 
            override_atom_type=override_atom_type,
            silent=silent_basis_analysis
        )
    else:
        molecule = load_molecule_with_override(
            idx=molecule_idx, 
            vnode=use_vnodes, 
            override_atom_type=override_atom_type,
            silent=silent_basis_analysis
        )
    
    load_time = time.time() - start_time
    if not silent_basis_analysis:
        print(f"   Load time: {load_time:.2f}s")
    
    # Get molecule ID for proper identification - handle list format
    raw_id = molecule.id if hasattr(molecule, 'id') and molecule.id is not None else f"idx_{molecule_idx}"
    if isinstance(raw_id, (list, tuple)) and len(raw_id) > 0:
        # Preserve the original string format of the first element
        molecule_id = str(raw_id[0])
    else:
        molecule_id = str(raw_id) if raw_id is not None else f"idx_{molecule_idx}"
    
    print(f"\n{'='*60}")
    print(f"PROCESSING MOLECULE {molecule_id}")
    print(f"Dataset index: {molecule_idx}")
    if not silent_basis_analysis:
        print(f"Basis set: {basis_set_name}")
        print(f"Augmentation: {use_augmentation} (β={beta})")
        print(f"Virtual nodes: {use_vnodes}")
        print(f"Atom type override: {override_atom_type}")
        print(f"Custom GTOs: {use_custom_gtos}")
        if use_custom_gtos:
            print(f"Custom GTO config: {custom_gto_config}")
    print(f"{'='*60}")
    
    # Step 2: Get atom centers and types for basis placement
    if not silent_basis_analysis:
        print("\n2. Extracting atom centers and types...")
    
    atom_coords, atom_types, center_info = get_atom_centers_and_types(
        molecule, 
        use_vnodes=use_vnodes, 
        override_atom_type=override_atom_type,
        silent=silent_basis_analysis
    )
    
    if not silent_basis_analysis:
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
                    if not silent_gpu:
                        print("   Warning: All GPUs excluded, falling back to CPU")
                    devices = ["cpu"]
                else:
                    devices = [f"cuda:{i}" for i in available_gpus]
                    if exclude_gpus and not silent_gpu:
                        print(f"   Excluded GPUs: {exclude_gpus}")
            else:
                devices = [f"cuda:{i}" for i in all_gpus]
        else:
            devices = ["cpu"]
    
    if not silent_gpu:
        print(f"   Using devices: {devices}")

    # Step 3: Construct GTO basis (standard or custom)
    if not silent_basis_analysis:
        print("\n3. Constructing GTO basis...")
    
    start_time = time.time()
    
    if use_custom_gtos:
        # Set default custom config if not provided
        if custom_gto_config is None:
            custom_gto_config = {
                'L_values': [0, 1, 2, 3],
                'exponent_ranges': [(1e-1, 1.0e+3)],
                'beta': 2.0,
                'n_exponents_per_L': None,
                'add_diversity': False,
                'random_seed': None
            }
        
        # Generate molecule-specific random seed if diversity is enabled
        diversity_config = custom_gto_config.copy()
        if diversity_config.get('add_diversity', False):
            if diversity_config.get('random_seed') is None:
                seed_base = hash(str(molecule_id)) ^ molecule_idx
                diversity_config['random_seed'] = abs(seed_base) % (2**31)
                if not silent_basis_analysis:
                    print(f"   Generated diversity seed: {diversity_config['random_seed']} for molecule {molecule_id}")
        
        # Validate custom config
        validate_custom_basis_parameters(**diversity_config)
        
        gto_dict = create_gto_basis_custom(
            atom_types=atom_types,
            atom_coords=atom_coords,
            **diversity_config,
            silent=silent_basis_analysis,
            silent_analysis=silent_basis_analysis
        )
        
        # Verify all atoms in this molecule have the same exponents (only if not silent)
        if not silent_basis_analysis:
            print(f"\n   Exponents by atom type for molecule {molecule_id}:")
            unique_types = torch.unique(atom_types).tolist()
            for atom_type in unique_types:
                type_str = str(atom_type)
                if type_str in gto_dict:
                    gto = gto_dict[type_str]
                    expos = gto.expos.cpu().numpy()
                    print(f"     Atom type {atom_type}: {len(expos)} exponents")
                    print(f"       Range: {expos.min():.3e} to {expos.max():.3e}")
                    print(f"       First 10: {[f'{e:.3e}' for e in expos[:10]]}")
                    if len(expos) > 10:
                        print(f"       Last 5:  {[f'{e:.3e}' for e in expos[-5:]]}")
            
            print(f"\n   Verifying exponent consistency for molecule {molecule_id}...")
            if len(unique_types) > 1:
                reference_expos = gto_dict[str(unique_types[0])].expos
                reference_Ls = gto_dict[str(unique_types[0])].Ls
                print(f"     Reference atom type: {unique_types[0]}")
                print(f"     Reference exponents: {reference_expos.shape[0]} total")
                
                for atom_type in unique_types[1:]:
                    other_expos = gto_dict[str(atom_type)].expos
                    other_Ls = gto_dict[str(atom_type)].Ls
                    
                    expos_match = torch.allclose(reference_expos, other_expos, rtol=1e-10)
                    Ls_match = torch.allclose(reference_Ls.float(), other_Ls.float(), rtol=1e-10)
                    
                    if expos_match and Ls_match:
                        print(f"     ✓ Atom type {atom_type}: identical exponents and L values")
                    else:
                        print(f"     ✗ ERROR: Atom type {atom_type} has different basis!")
                        if not expos_match:
                            print(f"       Exponent mismatch:")
                            print(f"         Reference: {reference_expos[:3].tolist()}")
                            print(f"         Different: {other_expos[:3].tolist()}")
                        if not Ls_match:
                            print(f"       L value mismatch:")
                            print(f"         Reference: {reference_Ls[:10].tolist()}")
                            print(f"         Different: {other_Ls[:10].tolist()}")
                        raise RuntimeError(f"Basis mismatch in molecule {molecule_id}!")
            else:
                print(f"     ✓ Single atom type {unique_types[0]} - consistency guaranteed")
        
        basis_type = 'custom'
    else:
        gto_dict = create_gto_basis(
            atom_types=atom_types,
            atom_coords=atom_coords,
            basis_set_name=basis_set_name,
            use_augmentation=use_augmentation,
            beta=beta,
            vnode_elem=vnode_elem,
            silent=silent_basis_analysis
        )
        
        basis_type = 'standard'
    
    basis_time = time.time() - start_time
    
    # Calculate total basis functions
    total_basis_funcs = 0
    for t in torch.unique(atom_types):
        t_str = str(int(t.item()))
        if t_str in gto_dict:
            n_atoms_of_type = int((atom_types == t).sum().item())
            total_basis_funcs += n_atoms_of_type * gto_dict[t_str].outdim
    
    if not silent_basis_analysis:
        print(f"   Total basis functions: {total_basis_funcs}")
        print(f"   Basis construction time: {basis_time:.2f}s")
        print(f"   Basis type: {basis_type}")
    
    # Step 4: Compute overlap integrals as 2D matrix
    if not silent_basis_analysis:
        print("\n4. Computing 2D overlap integrals...")
    
    start_time = time.time()
    # Request compressed_format so the function returns the packed per-exponent
    # dense matrix and a mapping that includes 'basis_indices_mapping'. This
    # avoids reshaping later and ensures consistency with the compression logic.
    dense_overlap, mapping_info = compute_overlap_integrals_2d_scdp(
        molecule, gto_dict, atom_coords, atom_types, max_probes_per_chunk,
        devices=devices, exclude_gpus=exclude_gpus,
        silent_gpu=silent_gpu, silent_mapping=silent_basis_analysis,
        compressed_format=True
    )
    overlap_time = time.time() - start_time

    if not silent_basis_analysis:
        print(f"   Overlap computation time: {overlap_time:.2f}s")
        if dense_overlap is None:
            print("   No overlap contributions found (dense overlap is None)")
        else:
            print(f"   Dense per-exponent overlap shape: {dense_overlap.shape}")

    # If detailed analysis requested, temporarily reconstruct the full 2D
    # overlap matrix from the dense per-exponent representation for display
    if show_overlap_analysis and not silent_basis_analysis and dense_overlap is not None:
        basis_info = mapping_info.get('basis_info', {})
        n_exps = basis_info.get('n_exponents')
        n_basis = basis_info.get('total_basis_functions')
        if n_exps and n_basis:
            overlap_2d_for_display = torch.zeros(n_exps, n_basis, dtype=dense_overlap.dtype)
            basis_indices_mapping = mapping_info.get('basis_indices_mapping', [])
            for exp_idx, basis_list in enumerate(basis_indices_mapping):
                if not basis_list:
                    continue
                overlap_2d_for_display[exp_idx, basis_list] = dense_overlap[exp_idx, : len(basis_list)]

            print_overlap_analysis_table(
                overlap_2d_for_display, mapping_info, atom_coords, atom_types,
                max_atoms_display=1, max_exponents_display=15,
                L_values_to_show=show_L_values
            )

    # Convert dense overlap to CPU and double precision for storage (if present)
    dense_overlap_storage = dense_overlap.double().cpu() if dense_overlap is not None else None

    # Create a serializable version of mapping_info using helper
    mapping_info_storage = build_mapping_storage(mapping_info)

    # Compute summary statistics for logging using only the actual non-padded values
    stats = compute_nonpadded_stats(dense_overlap_storage, mapping_info_storage.get('basis_indices_mapping'))
    overlap_integral_sum = stats['sum']
    overlap_integral_mean = stats['mean']
    overlap_integral_std = stats['std']
    overlap_integral_max = stats['max']
    overlap_integral_min = stats['min']
    
    # Only print overlap summary if not in silent mode
    if not silent_basis_analysis:
        print(f"\n   2D Overlap integrals matrix summary:")
        # Report shape: prefer original sparse shape if available, otherwise report dense-per-exponent shape
        shape_str = safe_shape_from_mapping(mapping_info_storage, dense_overlap_storage)
        print(f"     Shape: {shape_str}")
        print(f"     Sum: {overlap_integral_sum:.6e}")
        print(f"     Mean: {overlap_integral_mean:.6e}")
        print(f"     Std: {overlap_integral_std:.6e}")
        print(f"     Min/Max: {overlap_integral_min:.6e} / {overlap_integral_max:.6e}")
    
    # Step 5: Create CustomMolecule (v2) object with 2D overlap data
    if not silent_basis_analysis:
        print("\n5. Creating CustomMolecule object...")
    
    # Add dense overlap to mapping storage so helper can access it
    mapping_info_storage['dense_overlap_matrix'] = dense_overlap_storage

    # Build final CustomMolecule (v2) from compressed representation using helper
    custom_molecule = build_custom_molecule_from_compressed(
        atom_types=atom_types,
        atom_coords=atom_coords,
        molecule_id=molecule_id,
        n_atom=len(atom_types),
        basis_type=basis_type,
        mapping_info_storage=mapping_info_storage,
        mapping_info=mapping_info,
    )

    # Add additional metadata including molecule ID and dataset index and mapping
    metadata_update = {
        'molecule_id': molecule_id,
        'dataset_index': molecule_idx,
        'basis_set_name': basis_set_name if not use_custom_gtos else 'custom',
        'use_augmentation': use_augmentation,
        'beta': beta,
        'total_basis_functions': total_basis_funcs,
        'basis_type': basis_type,
        'use_custom_gtos': use_custom_gtos,
        'overlap_int_2d_statistics': {
            'sum': overlap_integral_sum,
            'mean': overlap_integral_mean,
            'std': overlap_integral_std,
            'min': overlap_integral_min,
            'max': overlap_integral_max,
            'n_exponents': mapping_info_storage.get('basis_info', {}).get('n_exponents'),
            'n_basis_functions': mapping_info_storage.get('basis_info', {}).get('total_basis_functions'),
            'shape': [
                mapping_info_storage.get('basis_info', {}).get('n_exponents'),
                mapping_info_storage.get('basis_info', {}).get('total_basis_functions')
            ] if mapping_info_storage.get('basis_info') else None
        },
        'processing_time': {
            'load_time': load_time,
            'basis_time': basis_time,
            'overlap_time': overlap_time
        },
        'devices_used': devices,
        'max_probes_per_chunk': max_probes_per_chunk,
        'center_info': center_info
    }
    
    if use_custom_gtos:
        metadata_update['custom_gto_config'] = diversity_config  # Store the actual config used

    # Include the full mapping info in metadata for downstream tools that
    # expect an "overlap_mapping_info" attribute.
    metadata_update['overlap_mapping_info'] = mapping_info_storage
    # Also include atom_basis_structure and basis_info for convenience
    metadata_update['atom_basis_structure'] = mapping_info_storage.get('atom_basis_structure')
    metadata_update['basis_info'] = mapping_info_storage.get('basis_info')

    # Update the object's metadata
    if custom_molecule.metadata is None:
        custom_molecule.metadata = {}
    custom_molecule.metadata.update(metadata_update)
    
    if not silent_basis_analysis:
        print(f"   CustomMolecule created for {molecule_id}: {custom_molecule}")
    
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
    save_interval: int = 10,
    use_custom_gtos: bool = False,
    custom_gto_config: Optional[Dict] = None,
    show_L_values: Optional[List[int]] = None,
    silent_basis_analysis: bool = False,
    silent_gpu: bool = False,
    show_overlap_analysis: bool = True,
    use_direct_indexing: bool = False
) -> Dict[int, str]:
    """
    Process a batch of molecules and save them as CustomMolecule pickle files.
    
    Args:
        silent_basis_analysis: If True, minimize basis analysis output
        silent_gpu: If True, minimize GPU computation output  
        show_overlap_analysis: If True, show detailed overlap analysis table
        use_direct_indexing: If True, use direct dataset indexing
        (other args same as before)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*80}")
    print(f"BATCH PROCESSING {len(molecule_indices)} MOLECULES")
    print(f"Output directory: {output_path}")
    if not silent_basis_analysis:
        print(f"Basis set: {basis_set_name}")
        print(f"Virtual nodes: {use_vnodes}")
        print(f"Atom type override: {override_atom_type}")
        print(f"Custom GTOs: {use_custom_gtos}")
        if use_custom_gtos:
            print(f"Custom GTO config: {custom_gto_config}")
    print(f"{'#'*80}")
    
    results = {}
    successful = 0
    failed = 0
    
    batch_start_time = time.time()
    
    # Create log file path for periodic updates
    log_file = output_path / "processing_log.csv"
    
    for i, mol_idx in enumerate(molecule_indices):
        # Try processing with progressively smaller chunk sizes if CUDA OOM occurs
        current_chunk_size = max_probes_per_chunk
        min_chunk_size = 5000  # Minimum chunk size to try
        chunk_reduction = 10000  # Reduce by this amount each retry
        
        molecule_processed = False
        attempt = 1
        max_attempts = max(1, (max_probes_per_chunk - min_chunk_size) // chunk_reduction + 1)
        
        while not molecule_processed and current_chunk_size >= min_chunk_size:
            try:
                if attempt > 1:
                    print(f"   Attempt {attempt}/{max_attempts} for molecule {mol_idx} with chunk size {current_chunk_size}")
                
                # Process molecule
                custom_mol = process_molecule_to_custom_data(
                    molecule_idx=mol_idx,
                    basis_set_name=basis_set_name,
                    use_augmentation=use_augmentation,
                    beta=beta,
                    use_vnodes=use_vnodes,
                    override_atom_type=override_atom_type,
                    vnode_elem=vnode_elem,
                    max_probes_per_chunk=current_chunk_size,  # Use current chunk size
                    device=device,
                    devices=devices,
                    exclude_gpus=exclude_gpus,
                    use_custom_gtos=use_custom_gtos,
                    custom_gto_config=custom_gto_config,
                    show_L_values=show_L_values,
                    silent_basis_analysis=silent_basis_analysis,
                    silent_gpu=silent_gpu,
                    show_overlap_analysis=show_overlap_analysis,
                    use_direct_indexing=use_direct_indexing
                )
                
                # Generate output filename using the dataset index to match final
                # naming convention: molecule_<6-digit-zero-padded-index>.pkl
                # This avoids a separate renaming step (see notebooks/fix_name.py)
                seq_id = int(mol_idx)
                filename = f"molecule_{seq_id:06d}.pkl"
                output_file = output_path / filename
                
                # Save to pickle
                custom_mol.save_pickle(output_file)
                
                results[mol_idx] = str(output_file)
                successful += 1
                molecule_processed = True
                
                # Log success message with chunk size info if retry was needed
                if attempt > 1:
                    print(f"   ✓ Saved to: {output_file} (succeeded with chunk size {current_chunk_size})")
                else:
                    print(f"   ✓ Saved to: {output_file}")
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if this is a CUDA out of memory error
                is_cuda_oom = any(keyword in error_str for keyword in [
                    'cuda out of memory', 'out of memory', 'cuda_out_of_memory', 
                    'cudaerroroutofmemory', 'runtime error: cuda out of memory'
                ])
                
                if is_cuda_oom and current_chunk_size > min_chunk_size:
                    # Reduce chunk size and try again
                    current_chunk_size = max(min_chunk_size, current_chunk_size - chunk_reduction)
                    attempt += 1
                    
                    print(f"   ⚠ CUDA OOM error for molecule {mol_idx}, retrying with chunk size {current_chunk_size}")
                    
                    # Clear CUDA cache before retry
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                    
                    continue  # Try again with smaller chunk size
                else:
                    # Either not a CUDA OOM error, or we've exhausted retry options
                    if is_cuda_oom:
                        print(f"   ✗ Failed to process molecule {mol_idx} due to CUDA OOM even with minimum chunk size {current_chunk_size}")
                    else:
                        print(f"   ✗ Failed to process molecule {mol_idx}: {e}")
                    
                    results[mol_idx] = f"FAILED: {e}"
                    failed += 1
                    molecule_processed = True  # Stop trying
        
        # If we exhausted all attempts without success
        if not molecule_processed:
            print(f"   ✗ Failed to process molecule {mol_idx} after {max_attempts} attempts")
            results[mol_idx] = f"FAILED: CUDA OOM - exhausted all chunk size reductions"
            failed += 1
        
        # Progress update and log saving every 20 molecules
        if (i + 1) % 20 == 0 or (i + 1) == len(molecule_indices):
            elapsed = time.time() - batch_start_time
            progress = (i + 1) / len(molecule_indices)
            estimated_total = elapsed / progress if progress > 0 else 0
            remaining = estimated_total - elapsed
            
            print(f"\n{'='*40}")
            print(f"PROGRESS: {i+1}/{len(molecule_indices)} ({progress*100:.1f}%)")
            print(f"Successful: {successful}, Failed: {failed}")
            print(f"Elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s")
            
            # Save log file every 20 molecules
            save_processing_log(results, str(log_file))
            print(f"Log updated: {log_file}")
            print(f"{'='*40}")
        
        # Clear memory periodically (every 20 molecules, same as log saving)
        if (i + 1) % 20 == 0:
            clear_memory()
    
    # Final progress summary
    print(f"\n{'='*40}")
    print(f"PROCESSING SUMMARY")
    print(f"Total molecules: {len(molecule_indices)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Elapsed time: {time.time() - batch_start_time:.2f}s")
    
    # Save final log
    save_processing_log(results, str(log_file))
    print(f"Final log saved: {log_file}")
    print(f"{'='*40}")
    
    return results

def process_full_dataset(
    output_dir: str,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
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
    save_interval: int = 10,
    use_custom_gtos: bool = False,
    custom_gto_config: Optional[Dict] = None,
    show_L_values: Optional[List[int]] = None
) -> Dict[int, str]:
    """
    Process the full dataset (or a range) with production settings.
    
    Args:
        start_idx: Starting dataset index
        end_idx: Ending dataset index (if None, process to end of dataset)
        (other args same as process_molecules_batch)
    
    Returns:
        Dictionary mapping dataset_idx to output file path
    """
    from overlap_pred.load_mol import get_dataset_size
    
    # Get dataset size and determine range
    dataset_size = get_dataset_size(vnode=use_vnodes)
    if end_idx is None:
        end_idx = dataset_size
    
    end_idx = min(end_idx, dataset_size)
    molecule_indices = list(range(start_idx, end_idx))
    
    print(f"\n{'#'*80}")
    print(f"FULL DATASET PROCESSING")
    print(f"Dataset size: {dataset_size}")
    print(f"Processing range: {start_idx} to {end_idx-1} ({len(molecule_indices)} molecules)")
    print(f"Output directory: {output_dir}")
    print(f"Basis set: {basis_set_name}")
    print(f"Virtual nodes: {use_vnodes}")
    print(f"Atom type override: {override_atom_type}")
    print(f"Custom GTOs: {use_custom_gtos}")
    if use_custom_gtos:
        print(f"Custom GTO config: {custom_gto_config}")
    print(f"{'#'*80}")
    
    # Process with production settings (minimal output)
    results = process_molecules_batch(
        molecule_indices=molecule_indices,
        output_dir=output_dir,
        basis_set_name=basis_set_name,
        use_augmentation=use_augmentation,
        beta=beta,
        use_vnodes=use_vnodes,
        override_atom_type=override_atom_type,
        vnode_elem=vnode_elem,
        max_probes_per_chunk=max_probes_per_chunk,
        device=device,
        devices=devices,
        exclude_gpus=exclude_gpus,
        save_interval=save_interval,
        use_custom_gtos=use_custom_gtos,
        custom_gto_config=custom_gto_config,
        show_L_values=show_L_values,
        silent_basis_analysis=True,  # Production mode
        silent_gpu=True,             # Production mode
        show_overlap_analysis=False,  # Production mode
        use_direct_indexing=True     # Use direct dataset indexing
    )
    
    # Note: Log is already saved within process_molecules_batch, no need to save again
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
        writer.writerow(['dataset_index', 'molecule_id', 'status', 'output_file'])
        
        for mol_idx, result in results.items():
            if result.startswith('FAILED:'):
                status = 'FAILED'
                output_file = result
                molecule_id = 'UNKNOWN'
            else:
                status = 'SUCCESS'
                output_file = result
                # Extract molecule ID from filename
                filename = Path(output_file).stem
                parts = filename.split('_')
                if len(parts) >= 2:
                    molecule_id = parts[1]  # molecule_{ID}_...
                else:
                    molecule_id = 'UNKNOWN'
            
            writer.writerow([mol_idx, molecule_id, status, output_file])
    
    print(f"Processing log saved to: {log_path}")

def read_missing_molecules_file(txt_file: str) -> List[int]:
    """
    Read molecule indices from a txt file (like missing_files.txt) and convert
    the 6-digit format to dataset indices.
    
    The txt file contains numbers like 012034, which need to be converted to
    dataset indices. The format is: GGGNNN where GGG is the group (batch) and
    NNN is the index within that group.
    
    For example:
    - 012034 -> group 12, index 34 -> dataset_index = 12 * 1000 + 34 = 12_34
    - 008401 -> group 8, index 401 -> dataset_index = 8 * 1000 + 401 = 8_401
    
    Args:
        txt_file: Path to the text file containing 6-digit molecule numbers
        
    Returns:
        List of dataset indices ready for processing
    """
    from pathlib import Path
    
    txt_path = Path(txt_file)
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing molecules file not found: {txt_file}")
    
    dataset_indices = []
    
    with open(txt_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Validate format - should be exactly 6 digits
            if not line.isdigit() or len(line) != 6:
                print(f"Warning: Invalid format on line {line_num}: '{line}' (expected 6 digits)")
                continue
            
            # Parse the 6-digit number: GGGNNN
            six_digit = int(line)
            dataset_index = six_digit - 1
            dataset_indices.append(dataset_index)

    #dataset_indices.sort()
    print(f"Read {len(dataset_indices)} molecule indices from {txt_file}")
    print(f"Range: {min(dataset_indices)} to {max(dataset_indices)}")
    print(f"First 10: {dataset_indices[:10]}")
    if len(dataset_indices) > 10:
        print(f"Last 10: {dataset_indices[-10:]}")
    
    return dataset_indices

def process_missing_molecules_from_file(
    txt_file: str,
    output_dir: str,
    basis_set_name: str = 'def2-QZVPP',
    use_augmentation: bool = True,
    beta: float = 2.0,
    use_vnodes: bool = True,
    override_atom_type: Optional[int] = None,
    vnode_elem: int = 1,
    max_probes_per_chunk: int = 50000,
    device: str = 'cpu',
    devices: Optional[List[str]] = None,
    exclude_gpus: Optional[List[int]] = None,
    use_custom_gtos: bool = False,
    custom_gto_config: Optional[Dict] = None,
    show_L_values: Optional[List[int]] = None
) -> Dict[int, str]:
    """
    Process molecules listed in a missing files txt file.
    
    Args:
        txt_file: Path to text file containing 6-digit molecule numbers
        output_dir: Directory to save the generated files
        (other args same as process_molecules_batch)
        
    Returns:
        Dictionary mapping dataset_idx to output file path
    """
    # Read molecule indices from file
    molecule_indices = read_missing_molecules_file(txt_file)
    
    if not molecule_indices:
        print("No valid molecule indices found in file.")
        return {}
    
    print(f"\nProcessing {len(molecule_indices)} missing molecules from {txt_file}")
    
    # Use the existing batch processing function
    results = process_molecules_batch(
        molecule_indices=molecule_indices,
        output_dir=output_dir,
        basis_set_name=basis_set_name,
        use_augmentation=use_augmentation,
        beta=beta,
        use_vnodes=use_vnodes,
        override_atom_type=override_atom_type,
        vnode_elem=vnode_elem,
        max_probes_per_chunk=max_probes_per_chunk,
        device=device,
        devices=devices,
        exclude_gpus=exclude_gpus,
        use_custom_gtos=use_custom_gtos,
        custom_gto_config=custom_gto_config,
        show_L_values=show_L_values,
        silent_basis_analysis=True,  # Production mode
        silent_gpu=True,             # Production mode
        show_overlap_analysis=False,  # Production mode
        use_direct_indexing=True     # Use direct dataset indexing
    )
    
    return results

def main():
    """Main function to process molecules and create CustomMolecule pickle files."""
    
    # Configuration
    config = {
        'basis_set_name': 'def2-QZVPP',
        'use_augmentation': True,
        'beta': 2.0,
        'use_vnodes': True,
        'override_atom_type': None, 
        'max_probes_per_chunk': 30000,
        'exclude_gpus': [1],
        'device': 'cpu',
        'use_custom_gtos': True,  # Enable custom GTOs for diversity testing
        'custom_gto_config': {
            'L_values': [0, 1, 2, 3, 4],
            'exponent_ranges': [(8.7e-2, 5.704e+3)],
            'beta': 2.0,
            'n_exponents_per_L': None,
            'add_diversity': True,  # Enable diversity
            'random_seed': None  # Will be auto-generated per molecule
        },
        'show_L_values': [0, 1, 2]  # Show only s, p, d orbitals in table
    }
    
    # Test with a small set of molecules first
    #test_molecules = [34075, 5, 343, 11797, 39941]
    test_molecules = np.random.choice(100000, size=20, replace=False).tolist()
    
    # Create timestamped output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    vnode_suffix = "_vnodes" if config['use_vnodes'] else ""
    override_suffix = f"_override{config['override_atom_type']}" if config['override_atom_type'] else ""
    custom_suffix = "_custom" if config['use_custom_gtos'] else ""
    basis_name = "custom" if config['use_custom_gtos'] else config['basis_set_name']
    output_dir = f"/export/data/hmichael/scdp/data/full_dataset_beta={config['custom_gto_config']['beta']}_{timestamp}"
    
    print(f"Starting molecule processing...")
    print(f"Configuration: {config}")
    print(f"Test molecule indices: {test_molecules}")
    
    # Process molecules
    results = process_molecules_batch(
        molecule_indices=test_molecules,
        output_dir=output_dir,
        **config
    )
    
    # Note: Log is already saved within process_molecules_batch
    # Save processing log
    # log_file = f"{output_dir}/processing_log.csv"
    # save_processing_log(results, log_file)
    
    # Test loading one of the saved files
    successful_files = [path for path in results.values() if not path.startswith('FAILED:')]
    if successful_files:
        test_file = successful_files[0]
        print(f"\n{'='*60}")
        print(f"TESTING PICKLE LOAD")
        print(f"Loading: {test_file}")
        
        try:
            loaded_mol = CustomMolecule.load_pickle(test_file)
            molecule_id = loaded_mol.metadata.get('molecule_id', 'UNKNOWN')
            dataset_idx = loaded_mol.metadata.get('dataset_index', 'UNKNOWN')
            print(f"✓ Successfully loaded molecule {molecule_id} (dataset index {dataset_idx}): {loaded_mol}")
            
            if loaded_mol.overlap_int_2d is not None:
                print(f"2D Overlap integrals shape: {loaded_mol.overlap_int_2d.shape}")
                print(f"2D Overlap integrals sum: {loaded_mol.overlap_int_2d.sum():.6e}")
                print(f"2D Overlap integrals mean: {loaded_mol.overlap_int_2d.mean():.6e}")

                # Test analysis methods and mapping info stored in metadata
                mapping = loaded_mol.metadata.get('overlap_mapping_info')
                if mapping is not None:
                    exp_vals = mapping.get('exponent_values')
                    try:
                        n_exps = len(exp_vals)
                    except Exception:
                        n_exps = None
                    print(f"Number of exponents: {n_exps}")
                    try:
                        exp_min = min(exp_vals)
                        exp_max = max(exp_vals)
                        print(f"Exponent range: {exp_min:.2e} to {exp_max:.2e}")
                    except Exception:
                        pass

                    # Test L-value analysis if available on the object
                    if hasattr(loaded_mol, 'analyze_overlap_by_L'):
                        try:
                            L_analysis = loaded_mol.analyze_overlap_by_L()
                            if L_analysis:
                                print("L-value analysis available for atoms:", list(L_analysis.keys()))
                        except Exception:
                            pass
            
            print(f"Metadata keys: {list(loaded_mol.metadata.keys())}")
            print(f"Basis info: {loaded_mol.metadata.get('basis_set_name', 'N/A')}")
            if 'overlap_int_2d_statistics' in loaded_mol.metadata:
                stats = loaded_mol.metadata['overlap_int_2d_statistics']
                print(f"2D Overlap integrals statistics: shape={stats['shape']}, sum={stats['sum']:.6e}")
            
            # Test the new properties method
            print("\nAll object properties:")
            loaded_mol.print_properties()
        except Exception as e:
            print(f"✗ Failed to load: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nProcessing completed!")
    print(f"Results saved to: {output_dir}")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create CustomMolecule pickle files with overlap integrals")
    parser.add_argument("--input_csv", type=str, default=None,
                       help="Path to CSV file containing molecule indices")
    parser.add_argument("--missing_file", type=str, default=None,
                       help="Path to txt file with 6-digit missing molecule numbers")
    parser.add_argument("--full_dataset", action="store_true",
                       help="Process the entire dataset")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Starting dataset index for full dataset processing")
    parser.add_argument("--end_idx", type=int, default=None,
                       help="Ending dataset index for full dataset processing")
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
    parser.add_argument("--use_custom_gtos", action="store_true",
                       help="Use custom GTO basis instead of standard")
    parser.add_argument("--show_L_values", type=int, nargs='+', default=None,
                       help="L values to show in overlap table (e.g., 0 1 2 for s p d)")
    parser.add_argument("--add_diversity", action="store_true",
                       help="Add diversity to custom GTO exponents")
    parser.add_argument("--diversity_seed", type=int, default=None,
                       help="Base seed for diversity (will be modified per molecule)")
    
    args = parser.parse_args()
    
    if args.full_dataset:
        # Process full dataset
        config = {
            'basis_set_name': args.basis_set,
            'use_augmentation': True,
            'beta': 2.0,
            'use_vnodes': args.use_vnodes,
            'override_atom_type': args.override_atom_type,
            'max_probes_per_chunk': args.max_chunk_size,
            'exclude_gpus': [0,1,2],
            'device': 'cpu',
            'use_custom_gtos': args.use_custom_gtos,
            'custom_gto_config': {
                'L_values': [0, 1, 2, 3, 4],
                'exponent_ranges': [(8.7e-2, 5.704e+3)],
                'beta': 2.0,
                'n_exponents_per_L': None,
                'add_diversity': args.add_diversity,
                'random_seed': args.diversity_seed
            },
            'show_L_values': [0, 1, 2]
        }
        
        # Output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            vnode_suffix = "_vnodes" if config['use_vnodes'] else ""
            override_suffix = f"_override{config['override_atom_type']}" if config['override_atom_type'] else ""
            custom_suffix = "_custom" if config['use_custom_gtos'] else ""
            basis_name = "custom" if config['use_custom_gtos'] else config['basis_set_name']
            output_dir = f"/export/data/hmichael/scdp/data/full_dataset_beta={config['custom_gto_config']['beta']}_{timestamp}"
        
        print(f"Processing full dataset from index {args.start_idx} to {args.end_idx or 'end'}")
        print(f"Output directory: {output_dir}")
        
        # Process full dataset
        results = process_full_dataset(
            output_dir=output_dir,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            **config
        )
        
        # Note: Log is already saved within process_molecules_batch
        # Save log
        # log_file = f"{output_dir}/processing_log.csv"
        # save_processing_log(results, log_file)
        
        print(f"Full dataset processing completed! Results in: {output_dir}")
    
    elif args.missing_file:
        # Process molecules from missing file
        config = {
            'basis_set_name': args.basis_set,
            'use_augmentation': True,
            'beta': 2.0,
            'use_vnodes': args.use_vnodes,
            'override_atom_type': args.override_atom_type,
            'max_probes_per_chunk': args.max_chunk_size,
            'exclude_gpus': [0, 1],
            'device': 'cpu',
            'use_custom_gtos': args.use_custom_gtos,
            'custom_gto_config': {
                'L_values': [0, 1, 2, 3, 4],
                'exponent_ranges': [(8.7e-2, 5.74e+3)],
                'beta': 2.0,
                'n_exponents_per_L': None,
                'add_diversity': args.add_diversity,
                'random_seed': args.diversity_seed
            },
            'show_L_values': args.show_L_values
        }
        
        # Output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            vnode_suffix = "_vnodes" if config['use_vnodes'] else ""
            override_suffix = f"_override{config['override_atom_type']}" if config['override_atom_type'] else ""
            custom_suffix = "_custom" if config['use_custom_gtos'] else ""
            missing_suffix = f"_missing_{Path(args.missing_file).stem}"
            basis_name = "custom" if config['use_custom_gtos'] else config['basis_set_name']
            output_dir = f"/export/data/hmichael/scdp/data/missing_{timestamp}"
        
        print(f"Processing missing molecules from: {args.missing_file}")
        print(f"Output directory: {output_dir}")
        if config['use_custom_gtos']:
            print(f"Custom GTO config: {config['custom_gto_config']}")
        
        # Process missing molecules
        results = process_missing_molecules_from_file(
            txt_file=args.missing_file,
            output_dir=output_dir,
            **config
        )
        
        print(f"Processing completed! Results in: {output_dir}")
        
    elif args.input_csv:
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
            'exclude_gpus': [4],
            'device': 'cpu',
            'use_custom_gtos': args.use_custom_gtos,
            'custom_gto_config': {
                'L_values': [0, 1, 2, 3, 4],
                'exponent_ranges': [(8.7e-2, 5.74e+3)],
                'beta': 2.0,
                'n_exponents_per_L': None,
                'add_diversity': args.add_diversity,
                'random_seed': args.diversity_seed
            },
            'show_L_values': [0, 1, 2]
        }
        
        # Output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            vnode_suffix = "_vnodes" if config['use_vnodes'] else ""
            override_suffix = f"_override{config['override_atom_type']}" if config['override_atom_type'] else ""
            custom_suffix = "_custom" if config['use_custom_gtos'] else ""
            input_suffix = f"_from_{Path(args.input_csv).stem}"
            basis_name = "custom" if config['use_custom_gtos'] else config['basis_set_name']
            output_dir = f"custom_molecules{vnode_suffix}{override_suffix}{custom_suffix}{input_suffix}_{basis_name}_{timestamp}"
        
        print(f"Processing {len(molecule_indices)} molecules from CSV")
        print(f"Output directory: {output_dir}")
        if config['use_custom_gtos']:
            print(f"Custom GTO config: {config['custom_gto_config']}")
        
        # Process molecules
        results = process_molecules_batch(
            molecule_indices=molecule_indices,
            output_dir=output_dir,
            **config
        )
        
        # Note: Log is already saved within process_molecules_batch
        # Save log
        # log_file = f"{output_dir}/processing_log.csv"
        # save_processing_log(results, log_file)
        
        print(f"Processing completed! Results in: {output_dir}")

    else:
        # Run with default test molecules
        results = main()

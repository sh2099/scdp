from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List
import numpy as np
import torch
import json
import pickle
from pathlib import Path


def _to_tensor(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    try:
        return torch.as_tensor(x)
    except Exception:
        return torch.tensor(np.array(x))


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.array(x)


@dataclass
class CustomMolecule:
    """
    Compressed version of CustomMolecule that stores dense overlap matrices
    instead of sparse ones for more efficient storage.
    """
    # core molecular properties
    atom_types: Optional[torch.Tensor] = None
    coords: Optional[torch.Tensor] = None
    batch: Optional[torch.Tensor] = None
    cell: Optional[torch.Tensor] = None
    is_vnode: Optional[torch.Tensor] = None
    node_attrs: Optional[torch.Tensor] = None
    ptr: Optional[torch.Tensor] = None
    
    # identifiers and metadata
    id: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # counts
    n_atom: Optional[int] = None
    n_vnode: Optional[int] = None
    
    # method information
    build_method: Optional[str] = None
    vnode_method: Optional[str] = None
    
    # overlap data
    overlap_int_2d: Optional[torch.Tensor] = None  # shape: (n_exponents, n_atoms * \sum^L(2l + 1))
    exponent_values: Optional[torch.Tensor] = None  # the actual exponent values
    

    def __repr__(self):
        dense_shape = self.overlap_int_2d.shape if self.overlap_int_2d is not None else None
        # Format ID properly
        display_id = self.id
        if isinstance(self.id, (list, tuple)) and len(self.id) > 0:
            display_id = str(self.id[0])
        elif self.id is not None:
            display_id = str(self.id)
        
        return (
            f"CompressedCustomMolecule(n_atom={self.n_atom}, n_probe={self.n_probe}, "
            f"n_vnode={self.n_vnode}, overlaps_shape={dense_shape}, "
            f"id={display_id}, build_method={self.build_method})"
        )

    def get_properties(self):
        """Return a dictionary of all properties and their types/shapes."""
        properties = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if value is None:
                properties[field_name] = "None"
            elif isinstance(value, torch.Tensor):
                properties[field_name] = f"Tensor{list(value.shape)} ({value.dtype})"
            elif isinstance(value, dict):
                properties[field_name] = f"Dict with {len(value)} keys: {list(value.keys())[:5]}{'...' if len(value) > 5 else ''}"
            elif isinstance(value, (list, tuple)):
                properties[field_name] = f"{type(value).__name__} of length {len(value)}"
            else:
                properties[field_name] = f"{type(value).__name__}: {str(value)[:50]}{'...' if len(str(value)) > 50 else ''}"
        return properties

    def print_properties(self):
        """Print all properties in a formatted table."""
        properties = self.get_properties()
        print(f"\nCompressedCustomMolecule Properties:")
        print("-" * 60)
        for name, description in properties.items():
            print(f"{name:<25}: {description}")
        print("-" * 60)

    def get_overlap_for_exponent(self, exponent_idx: int):
        """Get overlap values for a specific exponent index."""
        if self.overlap_int_2d is not None and exponent_idx < self.overlap_int_2d.shape[0]:
            # Get the row for this exponent
            row = self.overlap_int_2d[exponent_idx, :]
            return row
        return None
    
    def get_n_qtm_L(self, L: int):
        n_m_tot = [1, 4, 9, 16, 25, 36, 49]
        return n_m_tot[L] if L < len(n_m_tot) else None

    def get_max_L(self):
        """Get the maximum L quantum number across all atoms."""
        n_m_tot = [1, 4, 9, 16, 25, 36, 49]
        n_bfs_per_exp = self.overlap_int_2d.shape[1] if self.overlap_int_2d is not None else 0
        n_qtm_nums = (n_bfs_per_exp // len(self.atom_types) if self.atom_types is not None else 0)
        #print(n_qtm_nums)
        if int(n_qtm_nums) in n_m_tot:
            return n_m_tot.index(int(n_qtm_nums))
        return

    def get_overlap_for_atom_L_m(self, atom_idx: int, L: int, m: int):
        """Get overlap values for specific atom, L, m quantum numbers across all exponents."""
        if self.overlap_int_2d is None:
            return None

        if atom_idx > len(self.atom_types) - 1:
            print(f"Invalid atom index: {atom_idx}")
            return None

        if L < 0 or L > self.get_max_L():
            print(f"Invalid L value: {L}")
            return None
        if m < -L or m > L:
            print(f"Invalid m value: {m}")
            return None
        
        # Compute column index from atom_idx, L and m
        # For each atom there are sum up to L_max of (2l +1) basis functions
        if L > 0:
            col_idx = atom_idx * self.get_n_qtm_L(self.get_max_L()) + self.get_n_qtm_L(L-1) + (m + L)
        else: 
            col_idx = atom_idx * self.get_n_qtm_L(self.get_max_L()) + (m + L)
        return self.overlap_int_2d[:, col_idx]

    def print_overlap_table(self, max_atoms_display: int = 5, max_exponents_display: int = 10,
                                   L_values_to_show: Optional[list] = None):
        """
        Print detailed table showing overlap values organized by exponents and atoms with L,m breakdown.
        Uses the dense overlap matrix format with hierarchical column headers.
        """
        if self.overlap_int_2d is None or self.exponent_values is None:
            print("No overlap data available.")
            return
        
        if self.atom_types is None:
            print("No atom types available.")
            return
        
        max_L = self.get_max_L()
        if max_L is None:
            print("Could not determine maximum L value.")
            return
        
        # Limit what we display
        n_atoms_to_show = min(max_atoms_display, len(self.atom_types))
        n_exps_to_show = min(max_exponents_display, len(self.exponent_values))
        
        # Filter L values if specified
        L_values = list(range(max_L + 1))
        if L_values_to_show is not None:
            L_values = [L for L in L_values if L in L_values_to_show]
        
        # Calculate column widths
        exp_idx_width = 8
        exp_val_width = 12
        m_col_width = 12
        
        # Create the three header lines
        header1 = f"{'Exp_Idx':<{exp_idx_width}}{'Exponent':<{exp_val_width}}|"
        header2 = f"{'':<{exp_idx_width}}{'':<{exp_val_width}}|"
        header3 = f"{'':<{exp_idx_width}}{'':<{exp_val_width}}|"
        
        # Build headers for each atom
        for atom_idx in range(n_atoms_to_show):
            atom_type = self.atom_types[atom_idx].item()
            
            # Calculate total width for this atom
            total_m_values = sum(2*L + 1 for L in L_values)
            atom_width = total_m_values * m_col_width
            
            # Header 1: Atom label
            atom_label = f"Atom_{atom_idx}(Z={atom_type})"
            header1 += f"{atom_label:^{atom_width}}|"
            
            # Header 2: L value labels (s, p, d, etc.)
            header2_atom = ""
            for L in L_values:
                L_name = ['s', 'p', 'd', 'f', 'g', 'h'][L] if L < 6 else f'L{L}'
                L_width = (2*L + 1) * m_col_width
                header2_atom += f"{L_name:^{L_width}}|"
            header2 += header2_atom
            
            # Header 3: m value labels
            header3_atom = ""
            for L in L_values:
                for m in range(-L, L+1):
                    m_label = f"m={m}"
                    header3_atom += f"{m_label:<{m_col_width}}"
                header3_atom += "|"
            header3 += header3_atom
        
        # Print headers
        print(header1)
        print(header2)
        print(header3)
        
        # Create separator line
        separator = "-" * len(header1)
        print(separator)
        
        # Print data rows
        for exp_idx in range(n_exps_to_show):
            exp_val = self.exponent_values[exp_idx].item()
            
            # Start row with exponent info
            row = f"{exp_idx:<{exp_idx_width}}{exp_val:<{exp_val_width}.3e}|"
            
            # Add data for each atom
            for atom_idx in range(n_atoms_to_show):
                atom_data = ""
                
                for L in L_values:
                    for m in range(-L, L+1):
                        # Get overlap value for this specific (atom, L, m, exponent) combination
                        try:
                            overlap_column = self.get_overlap_for_atom_L_m(atom_idx, L, m)
                            if overlap_column is not None and exp_idx < len(overlap_column):
                                overlap_val = overlap_column[exp_idx].item()
                                atom_data += f"{overlap_val:<{m_col_width-1}.3e} "
                            else:
                                atom_data += f"{'N/A':<{m_col_width}} "
                        except Exception:
                            atom_data += f"{'---':<{m_col_width}} "
                    
                    # Add separator after each L block
                    atom_data += "|"
                
                row += atom_data
            
            print(row)
        
        # Print summary statistics
        print("\n" + "=" * 80)
        print("L-VALUE CONTRIBUTION SUMMARY")
        print("=" * 80)
        print(f"{'Atom_Idx':<10}{'L':<5}{'L_Name':<8}{'Total_Overlap':<15}{'Mean_Overlap':<15}{'Abs_Sum':<15}")
        print("-" * 75)
        
        for atom_idx in range(n_atoms_to_show):
            atom_type = self.atom_types[atom_idx].item()
            
            for L in L_values:
                L_name = ['s', 'p', 'd', 'f', 'g', 'h'][L] if L < 6 else f'L{L}'
                
                # Collect all overlap values for this L across all m and exponents
                L_overlaps = []
                for m in range(-L, L+1):
                    try:
                        overlap_column = self.get_overlap_for_atom_L_m(atom_idx, L, m)
                        if overlap_column is not None:
                            L_overlaps.extend(overlap_column.tolist())
                    except Exception:
                        continue
                
                if L_overlaps:
                    total_overlap = sum(L_overlaps)
                    mean_overlap = total_overlap / len(L_overlaps)
                    abs_sum = sum(abs(x) for x in L_overlaps)
                    
                    print(f"{atom_idx:<10}{L:<5}{L_name:<8}{total_overlap:<15.3e}{mean_overlap:<15.3e}{abs_sum:<15.3e}")
        
        # Print exponent summary
        print("\n" + "=" * 60)
        print("EXPONENT CONTRIBUTION SUMMARY")
        print("=" * 60)
        print(f"{'Exp_Idx':<10}{'Exponent':<15}{'Total_Sum':<15}{'Mean_Value':<15}{'Max_Abs':<15}")
        print("-" * 70)
        
        for exp_idx in range(min(n_exps_to_show, 20)):  # Show up to 20 exponents in summary
            exp_val = self.exponent_values[exp_idx].item()
            exp_row = self.overlap_int_2d[exp_idx, :]
            
            # Calculate statistics for this exponent
            total_sum = exp_row.sum().item()
            mean_val = exp_row.mean().item()
            max_abs = exp_row.abs().max().item()
            
            print(f"{exp_idx:<10}{exp_val:<15.3e}{total_sum:<15.3e}{mean_val:<15.3e}{max_abs:<15.3e}")
        
        if n_exps_to_show < len(self.exponent_values):
            remaining = len(self.exponent_values) - n_exps_to_show
            print(f"... and {remaining} more exponents not shown")
        
        print("=" * 120)

    def save_pickle(self, path):
        """Save the CompressedCustomMolecule object as a pickle file."""
        path = Path(path)
        with open(path, "wb") as fp:
            pickle.dump(self, fp)

    @classmethod
    def load_pickle(cls, path):
        """Load a CompressedCustomMolecule object from a pickle file."""
        path = Path(path)
        with open(path, "rb") as fp:
            return pickle.load(fp)

    def to_dict(self):
        """Return a JSON-serializable dict (tensors -> lists)."""
        d = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            
            if field_name == "metadata":
                d[field_name] = self.metadata
            elif field_name in ["atom_basis_structure", "basis_info"]:
                d[field_name] = value  # These should already be serializable
            elif field_name == "basis_indices_mapping":
                d[field_name] = value  # List of lists, already serializable
            elif field_name == "original_sparse_shape":
                d[field_name] = list(value) if value else None
            else:
                arr = _to_numpy(value)
                if arr is None:
                    d[field_name] = None
                else:
                    d[field_name] = arr.tolist() if isinstance(arr, np.ndarray) else arr
        return d

    def save_json(self, path):
        """Save as JSON (note: may be large due to dense matrix)."""
        with open(path, "w") as fp:
            json.dump(self.to_dict(), fp)

    @classmethod
    def load_json(cls, path):
        """Load from JSON."""
        with open(path, "r") as fp:
            d = json.load(fp)
        
        # Rehydrate tensors
        return cls(
            atom_types=_to_tensor(d.get("atom_types")),
            coords=_to_tensor(d.get("coords")),
            batch=_to_tensor(d.get("batch")),
            cell=_to_tensor(d.get("cell")),
            is_vnode=_to_tensor(d.get("is_vnode")),
            node_attrs=_to_tensor(d.get("node_attrs")),
            ptr=_to_tensor(d.get("ptr")),
            id=d.get("id"),
            metadata=d.get("metadata", {}),
            n_atom=d.get("n_atom"),
            n_probe=d.get("n_probe"),
            n_vnode=d.get("n_vnode"),
            build_method=d.get("build_method"),
            vnode_method=d.get("vnode_method"),
            overlap_int_2d=_to_tensor(d.get("overlap_int_2d")),
            basis_indices_mapping=d.get("basis_indices_mapping"),
            exponent_values=_to_tensor(d.get("exponent_values")),
            original_sparse_shape=tuple(d.get("original_sparse_shape")) if d.get("original_sparse_shape") else None,
            atom_basis_structure=d.get("atom_basis_structure"),
            basis_info=d.get("basis_info"),
        )

    @property
    def n_probe(self):
        """Calculate n_probe from overlap matrix if not set."""
        if hasattr(self, '_n_probe') and self._n_probe is not None:
            return self._n_probe
        # Calculate from overlap matrix dimensions if available
        if self.overlap_int_2d is not None and self.atom_types is not None:
            n_atoms = len(self.atom_types)
            total_basis_funcs = self.overlap_int_2d.shape[1]
            basis_per_atom = total_basis_funcs // n_atoms if n_atoms > 0 else 0
            # This is an approximation - actual n_probe calculation depends on the specific method
            return basis_per_atom * n_atoms
        return None
    
    @n_probe.setter
    def n_probe(self, value):
        self._n_probe = value

    @classmethod
    def from_compressed_custom_molecule(cls, compressed_mol):
        """
        Create CustomMolecule from a CompressedCustomMolecule instance.
        Converts the dense overlap matrix back to the 2D format used by CustomMolecule.
        """
        if compressed_mol.dense_overlap_matrix is None:
            # No overlap data to convert
            overlap_int_2d = None
            exponent_values = None
        else:
            overlap_int_2d = compressed_mol.dense_overlap_matrix
            exponent_values = compressed_mol.exponent_values
        
        # Create the new CustomMolecule instance
        custom_mol = cls(
            # Copy all molecular properties
            atom_types=compressed_mol.atom_types,
            coords=compressed_mol.coords,
            batch=compressed_mol.batch,
            cell=compressed_mol.cell,
            is_vnode=compressed_mol.is_vnode,
            node_attrs=compressed_mol.node_attrs,
            ptr=compressed_mol.ptr,
            id=compressed_mol.id,
            metadata=compressed_mol.metadata.copy() if compressed_mol.metadata else {},
            n_atom=compressed_mol.n_atom,
            n_vnode=compressed_mol.n_vnode,
            build_method=compressed_mol.build_method,
            vnode_method=compressed_mol.vnode_method,
            
            # Convert overlap data
            overlap_int_2d=overlap_int_2d,
            exponent_values=exponent_values,
        )
        
        # Set n_probe if available
        if hasattr(compressed_mol, 'n_probe') and compressed_mol.n_probe is not None:
            custom_mol.n_probe = compressed_mol.n_probe
        
        return custom_mol

    @classmethod
    def batch_convert_from_compressed(cls, input_dir, output_dir, pattern="*.pkl", 
                                    batch_size=100, max_workers=None, 
                                    skip_existing=True, validate_conversion=False, test=True):
        """
        Optimized batch conversion from CompressedCustomMolecule to CustomMolecule format.
        
        Args:
            input_dir: Directory containing CompressedCustomMolecule pickle files
            output_dir: Directory to save CustomMolecule files
            pattern: File pattern to match (default: "*.pkl")
            batch_size: Number of files to process in each batch
            max_workers: Maximum number of worker processes (None for auto)
            skip_existing: Skip files that already exist in output directory
            validate_conversion: Perform validation checks on converted molecules
        """
        from pathlib import Path
        import time
        import gc
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        from overlap_pred.compressed_custom_data import CompressedCustomMolecule
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all input files
        input_files = list(input_path.glob(pattern))
        if test:
            input_files = input_files[:10]
        print(f"Found {len(input_files)} files to convert")
        
        if skip_existing:
            # Filter out files that already exist
            existing_files = set(f.name for f in output_path.glob(pattern))
            input_files = [f for f in input_files if f.name not in existing_files]
            print(f"Skipping {len(existing_files)} existing files, {len(input_files)} remaining")
        
        if not input_files:
            print("No files to convert")
            return
        
        # Set up multiprocessing
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
        
        print(f"Using {max_workers} worker processes")
        print(f"Processing in batches of {batch_size}")
        
        # Calculate folder sizes
        def get_folder_size(folder_path):
            total_size = 0
            for file_path in Path(folder_path).glob(pattern):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        
        def format_size(size_bytes):
            if size_bytes == 0:
                return "0 B"
            size_names = ["B", "KB", "MB", "GB", "TB"]
            import math
            i = int(math.floor(math.log(size_bytes, 1024)))
            p = math.pow(1024, i)
            s = round(size_bytes / p, 2)
            return f"{s} {size_names[i]}"
        
        initial_size = get_folder_size(input_path)
        print(f"Initial compressed folder size: {format_size(initial_size)}")
        
        # Process files in batches
        total_processed = 0
        total_errors = 0
        start_time = time.time()
        
        for batch_start in range(0, len(input_files), batch_size):
            batch_end = min(batch_start + batch_size, len(input_files))
            batch_files = input_files[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(input_files) + batch_size - 1)//batch_size}")
            print(f"Files {batch_start + 1}-{batch_end} of {len(input_files)}")
            
            batch_start_time = time.time()
            
            # Process batch with multiprocessing
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all files in batch
                future_to_file = {
                    executor.submit(cls._convert_single_file, input_file, output_path, validate_conversion): input_file
                    for input_file in batch_files
                }
                
                # Collect results
                batch_processed = 0
                batch_errors = 0
                
                for future in as_completed(future_to_file):
                    input_file = future_to_file[future]
                    try:
                        success, error_msg = future.result()
                        if success:
                            batch_processed += 1
                        else:
                            batch_errors += 1
                            print(f"Error converting {input_file.name}: {error_msg}")
                    except Exception as e:
                        batch_errors += 1
                        print(f"Exception converting {input_file.name}: {e}")
            
            batch_time = time.time() - batch_start_time
            total_processed += batch_processed
            total_errors += batch_errors
            
            print(f"Batch completed: {batch_processed} successful, {batch_errors} errors")
            print(f"Batch time: {batch_time:.1f}s, Rate: {batch_processed/batch_time:.1f} files/s")
            
            # Progress update
            elapsed_time = time.time() - start_time
            files_remaining = len(input_files) - (batch_end)
            if total_processed > 0:
                estimated_total_time = elapsed_time * len(input_files) / (batch_end)
                estimated_remaining = estimated_total_time - elapsed_time
                print(f"Progress: {batch_end}/{len(input_files)} ({100*batch_end/len(input_files):.1f}%)")
                print(f"Estimated time remaining: {estimated_remaining/60:.1f} minutes")
            
            # Force garbage collection between batches
            gc.collect()
        
        # Final summary
        total_time = time.time() - start_time
        final_size = get_folder_size(output_path)
        
        print(f"\n" + "="*80)
        print("BATCH CONVERSION COMPLETE")
        print("="*80)
        print(f"Total files processed: {total_processed}")
        print(f"Total errors: {total_errors}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average rate: {total_processed/total_time:.1f} files/s")
        
        print(f"\nStorage comparison:")
        print(f"Compressed folder size: {format_size(initial_size)}")
        print(f"CustomMolecule folder size: {format_size(final_size)}")
        
        if initial_size > 0 and final_size > 0:
            size_ratio = final_size / initial_size
            print(f"Size ratio: {size_ratio:.1f}x ({'larger' if size_ratio > 1 else 'smaller'})")
        
        print("="*80)

    @staticmethod
    def _convert_single_file(input_file, output_dir, validate=False):
        """
        Convert a single file from compressed to CustomMolecule format.
        Used by batch processing with multiprocessing.
        
        Returns:
            tuple: (success: bool, error_message: str or None)
        """
        try:
            from overlap_pred.compressed_custom_data import CompressedCustomMolecule
            
            # Load compressed molecule
            compressed_mol = CompressedCustomMolecule.load_pickle(input_file)
            
            # Convert to CustomMolecule
            custom_mol = CustomMolecule.from_compressed_custom_molecule(compressed_mol)
            
            # Validation if requested
            if validate:
                validation_error = CustomMolecule._validate_conversion(compressed_mol, custom_mol)
                if validation_error:
                    return False, f"Validation failed: {validation_error}"
            
            # Save converted molecule
            output_file = output_dir / input_file.name
            custom_mol.save_pickle(output_file)
            
            return True, None
            
        except Exception as e:
            return False, str(e)

    @staticmethod
    def _validate_conversion(compressed_mol, custom_mol):
        """
        Validate that conversion from compressed to CustomMolecule was successful.
        
        Returns:
            str: Error message if validation fails, None if successful
        """
        try:
            # Check basic properties
            if compressed_mol.n_atom != custom_mol.n_atom:
                return f"n_atom mismatch: {compressed_mol.n_atom} vs {custom_mol.n_atom}"
            
            if compressed_mol.n_vnode != custom_mol.n_vnode:
                return f"n_vnode mismatch: {compressed_mol.n_vnode} vs {custom_mol.n_vnode}"
            
            # Check tensor shapes
            if (compressed_mol.atom_types is not None and custom_mol.atom_types is not None):
                if compressed_mol.atom_types.shape != custom_mol.atom_types.shape:
                    return f"atom_types shape mismatch"
            
            # Check overlap data
            if (compressed_mol.dense_overlap_matrix is not None and 
                custom_mol.overlap_int_2d is not None):
                
                # Should have same number of exponents
                if compressed_mol.dense_overlap_matrix.shape[0] != custom_mol.overlap_int_2d.shape[0]:
                    return f"overlap matrix exponent count mismatch"
                
                # Basic sum check (approximate due to format differences)
                import torch
                compressed_sum = compressed_mol.dense_overlap_matrix.sum().item()
                custom_sum = custom_mol.overlap_int_2d.sum().item()
                
                if abs(compressed_sum - custom_sum) > 1e-6:
                    return f"overlap data sum mismatch: {compressed_sum} vs {custom_sum}"
            
            return None  # Validation passed
            
        except Exception as e:
            return f"Validation exception: {e}"

    @classmethod
    def estimate_conversion_resources(cls, input_dir, pattern="*.pkl", sample_size=10):
        """
        Estimate memory and time requirements for batch conversion.
        
        Args:
            input_dir: Directory containing compressed files
            pattern: File pattern to match
            sample_size: Number of files to sample for estimation
        """
        from pathlib import Path
        import time
        import psutil
        import os
        from overlap_pred.compressed_custom_data import CompressedCustomMolecule
        
        input_path = Path(input_dir)
        input_files = list(input_path.glob(pattern))
        
        if not input_files:
            print("No files found for estimation")
            return
        
        # Sample files for estimation
        sample_files = input_files[:sample_size] if len(input_files) >= sample_size else input_files
        
        print(f"Estimating conversion requirements using {len(sample_files)} sample files...")
        
        # Memory usage before
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        total_time = 0
        peak_memory = initial_memory
        
        for i, sample_file in enumerate(sample_files):
            try:
                start_time = time.time()
                
                # Load and convert
                compressed_mol = CompressedCustomMolecule.load_pickle(sample_file)
                custom_mol = cls.from_compressed_custom_molecule(compressed_mol)
                
                conversion_time = time.time() - start_time
                total_time += conversion_time
                
                # Check memory usage
                current_memory = process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
                
                print(f"Sample {i+1}/{len(sample_files)}: {conversion_time:.3f}s, {current_memory:.1f}MB")
                
                # Clean up
                del compressed_mol, custom_mol
                
            except Exception as e:
                print(f"Error with sample {sample_file.name}: {e}")
        
        # Calculate estimates
        avg_time_per_file = total_time / len(sample_files)
        memory_per_file = (peak_memory - initial_memory) / len(sample_files)
        
        total_files = len(input_files)
        estimated_total_time = avg_time_per_file * total_files
        estimated_peak_memory = initial_memory + (memory_per_file * 8)  # Assume 8 parallel processes
        
        print(f"\n" + "="*60)
        print("CONVERSION RESOURCE ESTIMATION")
        print("="*60)
        print(f"Total files to convert: {total_files:,}")
        print(f"Average time per file: {avg_time_per_file:.3f}s")
        print(f"Estimated total time: {estimated_total_time/3600:.1f} hours")
        print(f"Memory per file: {memory_per_file:.1f}MB")
        print(f"Estimated peak memory (8 workers): {estimated_peak_memory:.1f}MB")
        
        # Recommendations
        available_memory = psutil.virtual_memory().available / 1024 / 1024
        print(f"Available system memory: {available_memory:.1f}MB")
        
        if estimated_peak_memory > available_memory * 0.8:
            recommended_workers = max(1, int(available_memory * 0.8 / (memory_per_file * 10)))
            print(f"⚠️  Recommended max workers: {recommended_workers} (to avoid memory issues)")
        else:
            print("✓ Memory usage should be acceptable with 8 workers")
        
        # Optimal batch size recommendation
        optimal_batch = max(50, min(500, int(1000 / avg_time_per_file)))
        print(f"Recommended batch size: {optimal_batch}")
        
        print("="*60)

def batch_convert_compressed_to_v2(input_dir, output_dir, **kwargs):
    """
    Convenience function for batch conversion from compressed to CustomMolecule v2.
    
    Args:
        input_dir: Directory containing CompressedCustomMolecule files
        output_dir: Directory to save CustomMolecule files
        **kwargs: Additional arguments passed to batch_convert_from_compressed
    """
    return CustomMolecule.batch_convert_from_compressed(input_dir, output_dir, **kwargs)



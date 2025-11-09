"""
New dataset implementation that treats each (molecule, exponent) pair as a separate data point.
"""

import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import glob
from pathlib import Path


class OverlapExponentDataset(Dataset):
    """
    Dataset that treats each (molecule, exponent) combination as a separate data point.
    
    For a molecule with N exponents, this creates N separate data points:
    - Input: atom_types, coords, single_exponent_value  
    - Target: overlap_row corresponding to that exponent
    
    This allows batching across different molecules and exponents.
    """
    
    def __init__(
        self,
        data_dir: str,
        subset_size: Optional[int] = None,
        transforms=None,
        cache_size: int = 1000,
        use_virtual_nodes: bool = True,
    ):
        """
        Args:
            data_dir: Directory containing molecule .pkl files
            subset_size: If provided, only use this many molecules (for testing)
            transforms: Optional transforms to apply to data
            cache_size: Number of molecules to keep in memory cache
            use_virtual_nodes: Whether to include virtual nodes in atom_types/coords
        """
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.cache_size = cache_size
        self.use_virtual_nodes = use_virtual_nodes
        
        # Find all molecule files
        self.molecule_files = sorted(list(self.data_dir.glob("molecule_*.pkl")))
        
        if subset_size is not None:
            self.molecule_files = self.molecule_files[:subset_size]
        
        print(f"Found {len(self.molecule_files)} molecule files")
        
        # Build index: each entry is (molecule_idx, exponent_idx)
        self.data_index = []
        self._build_index()
        
        # Cache for loaded molecules
        self._molecule_cache = {}
        self._cache_order = []
        
        print(f"Built index with {len(self.data_index)} data points")
        print(f"Average exponents per molecule: {len(self.data_index) / len(self.molecule_files):.1f}")
    
    def _build_index(self):
        """Build index of (molecule_idx, exponent_idx) pairs"""
        print("Building data index...")
        
        # Sample a few molecules to get typical number of exponents
        sample_size = min(10, len(self.molecule_files))
        exponent_counts = []
        
        for i in range(sample_size):
            try:
                with open(self.molecule_files[i], 'rb') as f:
                    mol_data = pickle.load(f)
                    n_exponents = len(mol_data.exponent_values)
                    exponent_counts.append(n_exponents)
            except Exception as e:
                print(f"Warning: Could not load {self.molecule_files[i]}: {e}")
                continue
        
        if not exponent_counts:
            raise RuntimeError("Could not load any sample molecules")
        
        # Check if all molecules have same number of exponents
        unique_counts = list(set(exponent_counts))
        if len(unique_counts) == 1:
            # All molecules have same number of exponents - fast path
            n_exponents = unique_counts[0]
            print(f"All molecules have {n_exponents} exponents")
            
            for mol_idx in range(len(self.molecule_files)):
                for exp_idx in range(n_exponents):
                    self.data_index.append((mol_idx, exp_idx))
        else:
            # Variable number of exponents - need to check each file
            print(f"Variable exponents per molecule: {unique_counts}")
            print("Scanning all files (this may take a while)...")
            
            for mol_idx, mol_file in enumerate(self.molecule_files):
                try:
                    with open(mol_file, 'rb') as f:
                        mol_data = pickle.load(f)
                        n_exponents = len(mol_data.exponent_values)
                        
                    for exp_idx in range(n_exponents):
                        self.data_index.append((mol_idx, exp_idx))
                        
                    if (mol_idx + 1) % 1000 == 0:
                        print(f"Processed {mol_idx + 1}/{len(self.molecule_files)} files")
                        
                except Exception as e:
                    print(f"Warning: Could not load {mol_file}: {e}")
                    continue
    
    def _load_molecule(self, mol_idx: int):
        """Load molecule data with caching"""
        if mol_idx in self._molecule_cache:
            return self._molecule_cache[mol_idx]
        
        # Load molecule
        mol_file = self.molecule_files[mol_idx]
        try:
            with open(mol_file, 'rb') as f:
                mol_data = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Could not load {mol_file}: {e}")
        
        # Add to cache
        self._molecule_cache[mol_idx] = mol_data
        self._cache_order.append(mol_idx)
        
        # Remove oldest if cache is full
        if len(self._cache_order) > self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._molecule_cache[oldest]
        
        return mol_data
    
    def __len__(self) -> int:
        return len(self.data_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single data point: (molecule, exponent) pair
        
        Returns:
            Dict containing:
                - atom_types: [N_nodes] atomic numbers (including virtual nodes if enabled)
                - coords: [N_nodes, 3] atomic coordinates  
                - exponent_value: scalar exponent value
                - target: [N_basis_functions] overlap row for this exponent
                - molecule_id: string identifier for the molecule
                - exponent_idx: index of this exponent within the molecule
        """
        mol_idx, exp_idx = self.data_index[idx]
        
        # Load molecule data
        mol_data = self._load_molecule(mol_idx)
        
        # Extract basic data
        atom_types = mol_data.atom_types.clone()
        coords = mol_data.coords.clone()
        exponent_value = mol_data.exponent_values[exp_idx].clone()
        overlap_row = mol_data.overlap_int_2d[exp_idx].clone()
        
        # Filter out virtual nodes if requested
        if not self.use_virtual_nodes:
            if hasattr(mol_data, 'is_vnode'):
                real_mask = ~mol_data.is_vnode
                atom_types = atom_types[real_mask]
                coords = coords[real_mask]
            else:
                # Assume virtual nodes have atom_type = 0
                real_mask = atom_types != 0
                atom_types = atom_types[real_mask]
                coords = coords[real_mask]
        
        # Create data dict
        data = {
            'atom_types': atom_types,
            'coords': coords,
            'exponent_value': exponent_value,
            'target': overlap_row,
            'molecule_id': getattr(mol_data, 'id', f'molecule_{mol_idx}'),
            'exponent_idx': torch.tensor(exp_idx, dtype=torch.long),
            'molecule_idx': torch.tensor(mol_idx, dtype=torch.long),
        }
        
        # Apply transforms if provided
        if self.transforms is not None:
            data = self.transforms(data)
        
        return data
    
    def get_molecule_info(self, mol_idx: int) -> Dict[str, Any]:
        """Get information about a specific molecule"""
        mol_data = self._load_molecule(mol_idx)
        
        return {
            'molecule_id': getattr(mol_data, 'id', f'molecule_{mol_idx}'),
            'n_atoms': len(mol_data.atom_types),
            'n_real_atoms': getattr(mol_data, 'n_atom', (mol_data.atom_types != 0).sum().item()),
            'n_virtual_nodes': getattr(mol_data, 'n_vnode', (mol_data.atom_types == 0).sum().item()),
            'n_exponents': len(mol_data.exponent_values),
            'exponent_range': [mol_data.exponent_values.min().item(), mol_data.exponent_values.max().item()],
            'overlap_shape': list(mol_data.overlap_int_2d.shape),
        }
    
    def get_dataset_stats(self, max_samples: int = 1000) -> Dict[str, Any]:
        """Get statistics about the dataset"""
        sample_indices = torch.randperm(len(self))[:max_samples].tolist()
        
        exponent_values = []
        overlap_values = []
        n_atoms_list = []
        
        for idx in sample_indices:
            data = self[idx]
            exponent_values.append(data['exponent_value'].item())
            overlap_values.extend(data['target'].tolist())
            n_atoms_list.append(len(data['atom_types']))
        
        exponent_values = torch.tensor(exponent_values)
        overlap_values = torch.tensor(overlap_values)
        n_atoms_list = torch.tensor(n_atoms_list)
        
        return {
            'n_data_points': len(self),
            'n_molecules': len(self.molecule_files),
            'exponent_stats': {
                'min': exponent_values.min().item(),
                'max': exponent_values.max().item(),
                'mean': exponent_values.mean().item(),
                'std': exponent_values.std().item(),
            },
            'overlap_stats': {
                'min': overlap_values.min().item(),
                'max': overlap_values.max().item(),
                'mean': overlap_values.mean().item(),
                'std': overlap_values.std().item(),
            },
            'atoms_per_molecule': {
                'min': n_atoms_list.min().item(),
                'max': n_atoms_list.max().item(),
                'mean': n_atoms_list.float().mean().item(),
            }
        }


def collate_overlap_data(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching overlap data.
    
    Handles variable number of atoms per molecule by creating batch indices.
    For targets with variable lengths, we keep them as a list rather than
    trying to stack them - similar to how PyTorch Geometric handles variable-sized data.
    
    Also handles graph structure fields (edge_index, edge_distance_vec, etc.) that 
    are added by transforms.
    """
    batch_atom_types = []
    batch_coords = []
    batch_exponent_values = []
    batch_targets = []
    batch_indices = []
    batch_molecule_ids = []
    batch_exponent_indices = []
    batch_molecule_indices = []
    
    # Graph structure fields (may be present if transforms were applied)
    batch_edge_indices = []
    batch_edge_distance_vecs = []
    batch_edge_distances = []
    batch_shifts = []
    
    # Track cumulative atom offset for edge index adjustment
    atom_offset = 0
    
    # Track target lengths
    target_lengths = []
    
    for i, data in enumerate(batch):
        batch_atom_types.append(data['atom_types'])
        batch_coords.append(data['coords'])
        batch_exponent_values.append(data['exponent_value'])
        batch_targets.append(data['target'])
        batch_molecule_ids.append(data['molecule_id'])
        batch_exponent_indices.append(data['exponent_idx'])
        batch_molecule_indices.append(data['molecule_idx'])
        
        # Create batch indices for this molecule's atoms
        n_atoms = len(data['atom_types'])
        batch_indices.extend([i] * n_atoms)
        
        target_lengths.append(len(data['target']))
        
        # Handle graph structure if present (from transforms)
        if 'edge_index' in data:
            # Adjust edge indices to account for batching
            # We need to add the atom_offset to both source and target indices
            edge_index = data['edge_index'].clone()
            edge_index[0] += atom_offset  # source indices
            edge_index[1] += atom_offset  # target indices
            batch_edge_indices.append(edge_index)
            
            # Other edge-related fields don't need offset adjustment
            if 'edge_distance_vec' in data:
                batch_edge_distance_vecs.append(data['edge_distance_vec'])
            if 'edge_distance' in data:
                batch_edge_distances.append(data['edge_distance'])
            if 'shifts' in data:
                batch_shifts.append(data['shifts'])
        
        atom_offset += n_atoms
    
    # Check if all targets have the same length
    if len(set(target_lengths)) == 1:
        # All same length - can stack normally
        targets = torch.stack(batch_targets)
    else:
        # Different lengths - keep as list like PyTorch Geometric does
        # This avoids the memory overhead of padding and is more flexible
        targets = batch_targets
        # Note: Commented out to reduce log verbosity during training
        # print(f"Note: Variable target lengths {target_lengths}, keeping as list")
    
    # Build output dictionary
    result = {
        'atom_types': torch.cat(batch_atom_types, dim=0),
        'coords': torch.cat(batch_coords, dim=0),
        'exponent_value': torch.stack(batch_exponent_values),
        'target': targets,  # Can be either stacked tensor or list of tensors
        'batch': torch.tensor(batch_indices, dtype=torch.long),
        'molecule_id': batch_molecule_ids,
        'exponent_idx': torch.stack(batch_exponent_indices),
        'molecule_idx': torch.stack(batch_molecule_indices),
        'target_lengths': torch.tensor(target_lengths, dtype=torch.long),
    }
    
    # Add graph structure fields if they were present
    if batch_edge_indices:
        result['edge_index'] = torch.cat(batch_edge_indices, dim=1)
    if batch_edge_distance_vecs:
        result['edge_distance_vec'] = torch.cat(batch_edge_distance_vecs, dim=0)
    if batch_edge_distances:
        result['edge_distance'] = torch.cat(batch_edge_distances, dim=0)
    if batch_shifts:
        result['shifts'] = torch.cat(batch_shifts, dim=0)
    
    return result


def compute_loss_from_list_targets(predictions: torch.Tensor, targets, loss_fn) -> torch.Tensor:
    """
    Helper function to compute loss when targets is a list of variable-sized tensors.
    
    Args:
        predictions: [batch_size, max_output_size] model predictions
        targets: List of target tensors with variable sizes
        loss_fn: Loss function (e.g., nn.MSELoss())
    
    Returns:
        Average loss across all targets
    """
    losses = []
    
    for i, target in enumerate(targets):
        # Extract prediction for this sample and ensure float dtype
        pred = predictions[i, :len(target)].float()  # Truncate to target size
        target = target.float()  # Ensure target is float
        
        # Compute loss for this sample
        loss = loss_fn(pred, target)
        losses.append(loss)
    
    # Return average loss
    return torch.stack(losses).mean()


if __name__ == "__main__":
    # Test the dataset
    data_dir = "/export/data/hmichael/scdp/data/full_comp_new"
    
    print("Testing OverlapExponentDataset...")
    dataset = OverlapExponentDataset(data_dir, subset_size=5, use_virtual_nodes=True)
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test getting individual samples
    print("\nTesting individual samples:")
    for i in range(3):
        sample = dataset[i]
        print(f"Sample {i}:")
        print(f"  Molecule: {sample['molecule_id']}")
        print(f"  Exponent idx: {sample['exponent_idx'].item()}")
        print(f"  Exponent value: {sample['exponent_value'].item():.6f}")
        print(f"  Atom types: {sample['atom_types']}")
        print(f"  Coords shape: {sample['coords'].shape}")
        print(f"  Target shape: {sample['target'].shape}")
    
    # Test dataloader with batching
    print("\nTesting DataLoader:")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_overlap_data)
    
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Batch size: {len(batch['molecule_id'])}")
        print(f"  Total atoms: {len(batch['atom_types'])}")
        print(f"  Exponent values: {batch['exponent_value']}")
        if isinstance(batch['target'], list):
            print(f"  Target (list): {[t.shape for t in batch['target']]}")
        else:
            print(f"  Target shape: {batch['target'].shape}")
        print(f"  Batch indices: {batch['batch']}")
        if i >= 2:  # Only show first few batches
            break
    
    # Get dataset statistics
    print("\nDataset statistics:")
    stats = dataset.get_dataset_stats(max_samples=100)
    for key, value in stats.items():
        print(f"  {key}: {value}")

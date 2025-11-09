"""
Data loading utilities for overlap prediction.

Handles loading and batching of CustomMolecule objects (new format) for training
the overlap prediction model. Each batch contains all exponents for a single molecule.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
from typing import List, Optional, Union, Dict, Any
import numpy as np
import sys
sys.path.append('/export/home/hmichael/scdp/overlap_pred')

from scdp.common.utils import get_edge_vectors_and_lengths
from scdp.model.overlap.transforms import (
    create_overlap_transforms, 
    create_overlap_transforms_from_config,
    create_overlap_data
)


class OverlapDataset(Dataset):
    """
    Dataset for loading CustomMolecule objects for overlap prediction.
    
    Each sample represents a complete molecule with all its exponents.
    Each batch will contain all (molecule, exponent) pairs for a single molecule.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path, List[str]] = None,
        cutoff: float = 6.0,
        max_neighbors: int = 50,
        transform=None,
        filter_fn=None,
        max_samples_per_molecule: Optional[int] = None,  # Limit exponents per molecule
        device: Optional[torch.device] = None,
        molecules: Optional[List] = None,  # For direct molecule input
        exponent_values: Optional[np.ndarray] = None,  # For external exponent values
    ):
        """
        Args:
            data_path: Path to pickle files or list of paths (used if molecules is None)
            cutoff: Cutoff distance for neighbor search
            max_neighbors: Maximum number of neighbors per atom
            transform: Optional transform to apply to molecules
            filter_fn: Optional function to filter molecules
            max_samples_per_molecule: Maximum number of exponents to use per molecule
            device: Device to load tensors on
            molecules: List of CustomMolecule objects (if provided, data_path is ignored)
            exponent_values: External exponent values to use for all molecules
        """
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.transform = transform
        self.filter_fn = filter_fn
        self.max_samples_per_molecule = max_samples_per_molecule
        self.device = device
        self.external_exponent_values = exponent_values
        
        if molecules is not None:
            # Use provided molecules directly
            self.molecules = molecules
            self.molecule_paths = None
        else:
            # Load from files
            self.molecules = None
            if data_path is not None:
                self.molecule_paths = self._collect_paths(data_path)
            else:
                self.molecule_paths = []
            
        # Create molecule samples (one per molecule, not per exponent)
        self.molecule_samples = self._create_molecule_samples()
        
        print(f"Loaded {len(self.molecule_samples)} molecules")
        if len(self.molecule_samples) > 0:
            # Count total exponent pairs for info
            total_pairs = sum(len(self._get_exponent_indices(i)) for i in range(len(self.molecule_samples)))
            print(f"Created {total_pairs} (molecule, exponent) pairs")
    
    def _collect_paths(self, data_path):
        """Collect all molecule pickle file paths."""
        if isinstance(data_path, list):
            paths = []
            for path in data_path:
                paths.extend(self._collect_paths(path))
            return paths
        
        data_path = Path(data_path)
        
        if data_path.is_file():
            return [data_path]
        elif data_path.is_dir():
            return list(data_path.glob("*.pkl"))
        else:
            raise ValueError(f"Invalid data path: {data_path}")
    
    def _create_molecule_samples(self):
        """Create molecule samples (one per molecule)."""
        if self.molecules is not None:
            # Use provided molecules
            return list(range(len(self.molecules)))
        else:
            # Use molecule paths
            valid_indices = []
            for i, mol_path in enumerate(self.molecule_paths):
                try:
                    # Quick check to see if molecule loads
                    with open(mol_path, 'rb') as f:
                        mol = pickle.load(f)
                    
                    # Apply filter if provided
                    if self.filter_fn and not self.filter_fn(mol):
                        continue
                        
                    valid_indices.append(i)
                except Exception as e:
                    print(f"Error loading {mol_path}: {e}")
                    continue
            return valid_indices
    
    def _get_molecule(self, mol_idx):
        """Get molecule by index."""
        if self.molecules is not None:
            return self.molecules[mol_idx]
        else:
            mol_path = self.molecule_paths[self.molecule_samples[mol_idx]]
            with open(mol_path, 'rb') as f:
                return pickle.load(f)
    
    def _get_exponent_indices(self, mol_idx):
        """Get list of exponent indices for a molecule."""
        mol = self._get_molecule(mol_idx)
        
        if self.external_exponent_values is not None:
            # Use external exponent values
            n_exponents = len(self.external_exponent_values)
        elif mol.exponent_values is not None:
            n_exponents = len(mol.exponent_values)
        else:
            # No exponent data, use single default
            n_exponents = 1
        
        # Limit exponents per molecule if specified
        if self.max_samples_per_molecule:
            n_exponents = min(n_exponents, self.max_samples_per_molecule)
        
        return list(range(n_exponents))
    
    def __len__(self):
        """Return number of molecules (not number of exponent pairs)."""
        return len(self.molecule_samples)
    
    def __getitem__(self, idx):
        """
        Load and process all (molecule, exponent) pairs for a single molecule.
        Returns a list of data dictionaries, one for each exponent.
        """
        try:
            mol = self._get_molecule(idx)
            exponent_indices = self._get_exponent_indices(idx)
            
            # Create data for each exponent
            batch_data = []
            for exp_idx in exponent_indices:
                if self.transform:
                    data = create_overlap_data(mol, exp_idx, self.transform)
                else:
                    # Fallback processing
                    data = self._process_molecule(mol, exp_idx)
                
                # Handle external exponent values
                if self.external_exponent_values is not None:
                    data['exponent_value'] = torch.tensor([self.external_exponent_values[exp_idx]], dtype=torch.float32)
                
                batch_data.append(data)
            
            return batch_data
            
        except Exception as e:
            print(f"Error processing molecule {idx}: {e}")
            return None
    
    def _process_molecule(self, mol, exp_idx):
        """
        Process a CustomMolecule + exponent index into model input format (fallback method).
        This is used when no transform is provided.
        """
        # Basic data
        data = {
            'atom_types': torch.from_numpy(mol.atom_types).long() if isinstance(mol.atom_types, np.ndarray) else mol.atom_types,
            'coords': torch.from_numpy(mol.coords).float() if isinstance(mol.coords, np.ndarray) else mol.coords,
            'mol_id': mol.id,
            'n_atom': mol.n_atom,
            'n_vnode': mol.n_vnode,
            'exponent_index': exp_idx,
            'original_molecule': mol
        }
        
        # Add exponent value
        if self.external_exponent_values is not None:
            data['exponent_value'] = torch.tensor([self.external_exponent_values[exp_idx]], dtype=torch.float32)
        elif mol.exponent_values is not None and exp_idx < len(mol.exponent_values):
            data['exponent_value'] = torch.tensor([mol.exponent_values[exp_idx]], dtype=torch.float32)
        else:
            data['exponent_value'] = torch.tensor([1.0], dtype=torch.float32)  # Default
        
        # Add overlap target if available
        if hasattr(mol, 'overlap_int_2d') and mol.overlap_int_2d is not None:
            data['overlap_target'] = mol.overlap_int_2d
        else:
            # Create dummy target for testing
            data['overlap_target'] = torch.zeros((mol.n_atom, 100))
        
        # Add edges using fallback method
        data = self._add_edges(data)
        
        return data
    
    def _add_edges(self, data):
        """Add edge information using radius graph (fallback method)."""
        coords = data['coords']
        n_atoms = len(coords)
        
        # Compute pairwise distances
        distances = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
        
        # Find edges within cutoff
        edge_mask = (distances <= self.cutoff) & (distances > 0)
        edge_indices = torch.where(edge_mask)
        
        if len(edge_indices[0]) > 0:
            edge_index = torch.stack(edge_indices, dim=0)
            
            # Limit neighbors if needed
            edge_index = self._limit_neighbors(edge_index, n_atoms)
            
            # Compute edge vectors and distances
            src_coords = coords[edge_index[0]]
            dst_coords = coords[edge_index[1]]
            edge_vectors = dst_coords - src_coords
            edge_distances = torch.norm(edge_vectors, dim=1)
            
            data['edge_index'] = edge_index
            data['edge_distance_vec'] = edge_vectors
            data['edge_distance'] = edge_distances
            data['shifts'] = torch.zeros((len(edge_distances), 3))  # No PBC shifts for now
        else:
            # No edges found
            data['edge_index'] = torch.empty((2, 0), dtype=torch.long)
            data['edge_distance_vec'] = torch.empty((0, 3))
            data['edge_distance'] = torch.empty((0,))
            data['shifts'] = torch.empty((0, 3))
        
        # Add batch indices (single molecule)
        data['batch'] = torch.zeros(n_atoms, dtype=torch.long)
        
        return data
    
    def _limit_neighbors(self, edge_indices, n_atoms):
        """Limit the number of neighbors per atom."""
        if self.max_neighbors is None:
            return edge_indices
        
        # Group edges by source atom
        limited_edges = []
        for atom_idx in range(n_atoms):
            # Find edges starting from this atom
            mask = edge_indices[0] == atom_idx
            atom_edges = edge_indices[:, mask]
            
            if atom_edges.shape[1] > self.max_neighbors:
                # Limit to max_neighbors
                atom_edges = atom_edges[:, :self.max_neighbors]
            
            if atom_edges.shape[1] > 0:
                limited_edges.append(atom_edges)
        
        if limited_edges:
            return torch.cat(limited_edges, dim=1)
        else:
            return torch.empty((2, 0), dtype=torch.long)


def overlap_collate_fn(batch):
    """
    Custom collate function for overlap prediction.
    
    Input: List of molecule batches, where each molecule batch is a list of 
           (molecule, exponent) data dictionaries for that molecule.
    
    The key insight: Each molecule forms its own batch! We don't mix molecules.
    So if we have 2 molecules in the DataLoader batch, we need to process them separately.
    """
    if len(batch) == 1:
        # Single molecule - this is the common case we want
        molecule_batch = batch[0]
        if molecule_batch is None:
            return None
        
        # Process all exponents for this single molecule
        return _collate_single_molecule(molecule_batch)
    else:
        # Multiple molecules - we need to return a list of batched molecules
        # This case should be rare since we want batch_size=1 for molecules
        return [_collate_single_molecule(mol_batch) for mol_batch in batch if mol_batch is not None]


def _collate_single_molecule(molecule_data_list):
    """
    Collate all exponents for a single molecule.
    
    Args:
        molecule_data_list: List of data dictionaries, one per exponent for this molecule
    
    Returns:
        Batched dictionary where batch dimension corresponds to exponents
    """
    if not molecule_data_list:
        return None
    
    # All items should have the same molecular structure
    first_item = molecule_data_list[0]
    n_exponents = len(molecule_data_list)
    
    # Start with molecular structure (same for all exponents)
    batched_data = {
        'atom_types': first_item['atom_types'],  # Same for all exponents
        'coords': first_item['coords'],          # Same for all exponents
    }
    
    # Add graph structure (same for all exponents)
    if 'edge_index' in first_item:
        batched_data['edge_index'] = first_item['edge_index']
    if 'edge_distance_vec' in first_item:
        batched_data['edge_distance_vec'] = first_item['edge_distance_vec']
    if 'edge_distance' in first_item:
        batched_data['edge_distance'] = first_item['edge_distance']
    if 'shifts' in first_item:
        batched_data['shifts'] = first_item['shifts']
    
    # Stack exponent-specific data
    batched_data['exponent_value'] = torch.stack([item['exponent_value'] for item in molecule_data_list])
    
    # Handle overlap targets (these vary by exponent)
    if 'overlap_target' in first_item:
        # Check if we need to select the right slice for each exponent
        overlap_targets = []
        for i, item in enumerate(molecule_data_list):
            target = item['overlap_target']
            if target.dim() >= 2 and target.shape[0] > 1:
                # Select the slice corresponding to this exponent
                overlap_targets.append(target[i:i+1])  # Keep batch dimension
            else:
                overlap_targets.append(target)
        
        batched_data['overlap_targets'] = overlap_targets
    
    # Create batch indices for atoms (each exponent gets its own batch index)
    n_atoms = len(batched_data['atom_types'])
    batch_indices = torch.arange(n_exponents).repeat_interleave(n_atoms)
    batched_data['batch'] = batch_indices
    
    # Repeat molecular data for each exponent for GNN processing
    batched_data['atom_types'] = batched_data['atom_types'].repeat(n_exponents)
    batched_data['coords'] = batched_data['coords'].repeat(n_exponents, 1)
    
    # Adjust edge indices for repeated molecular structure
    if 'edge_index' in batched_data and len(batched_data['edge_index'][0]) > 0:
        edge_offsets = torch.arange(n_exponents).unsqueeze(1) * n_atoms
        edge_indices_expanded = []
        
        # Repeat edge structure for each exponent
        for i in range(n_exponents):
            offset_edges = batched_data['edge_index'] + edge_offsets[i]
            edge_indices_expanded.append(offset_edges)
        
        batched_data['edge_index'] = torch.cat(edge_indices_expanded, dim=1)
        
        # Repeat edge features for each exponent
        if 'edge_distance_vec' in batched_data:
            batched_data['edge_distance_vec'] = batched_data['edge_distance_vec'].repeat(n_exponents, 1)
        if 'edge_distance' in batched_data:
            edge_dist = batched_data['edge_distance']
            if edge_dist.dim() == 0:  # Scalar tensor, add dimension
                edge_dist = edge_dist.unsqueeze(0)
            batched_data['edge_distance'] = edge_dist.repeat(n_exponents)
        if 'shifts' in batched_data:
            batched_data['shifts'] = batched_data['shifts'].repeat(n_exponents, 1)
    
    # Repeat exponent values to match atom dimension for embedding
    batched_data['exponent_value'] = batched_data['exponent_value'].repeat_interleave(n_atoms)
    
    # Metadata
    batched_data['mol_ids'] = [item['mol_id'] for item in molecule_data_list]
    batched_data['exp_indices'] = [item.get('exponent_index', i) for i, item in enumerate(molecule_data_list)]
    batched_data['n_exponents'] = n_exponents
    batched_data['n_atoms'] = n_atoms
    batched_data['n_total_atoms'] = n_atoms * n_exponents  # For debugging
    
    return batched_data


def create_overlap_dataloader(
    data_path,
    batch_size=1,  # Changed default to 1 since each molecule is its own batch
    cutoff=6.0,
    max_neighbors=50,
    max_samples_per_molecule=None,
    num_workers=4,
    shuffle=True,
    transform_config_path=None
):
    """
    Create a DataLoader for overlap prediction.
    
    Args:
        data_path: Path to the dataset directory
        batch_size: Should typically be 1 since each molecule forms its own batch
        cutoff: Cutoff distance for edge creation
        max_neighbors: Maximum number of neighbors per atom
        max_samples_per_molecule: Maximum samples per molecule (None for all)
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the data
        transform_config_path: Path to transform configuration file
    
    Returns:
        DataLoader for overlap prediction
    """
    # Create transforms based on config if provided
    transform = None
    if transform_config_path and Path(transform_config_path).exists():
        transform = create_overlap_transforms_from_config(transform_config_path)
    else:
        # Use default transforms
        transform = create_overlap_transforms_from_config(
            config_path=None,
            radius=cutoff,
            max_neighbors=max_neighbors
        )
    
    dataset = OverlapDataset(
        data_path=data_path,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        max_samples_per_molecule=max_samples_per_molecule,
        transform=transform
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,  # Typically 1 for molecule-level batching
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=overlap_collate_fn
    )


def create_dataset(molecules, exponent_values=None, use_transforms=True, config_path=None):
    """
    Create OverlapDataset from molecules and exponent values.
    
    Args:
        molecules: List of CustomMolecule objects
        exponent_values: Array of exponent values to use. If None, uses molecule's exponent_values
        use_transforms: Whether to apply transforms for ESCN compatibility
        config_path: Path to transform configuration file. If None, uses defaults.
    
    Returns:
        OverlapDataset
    """
    # Create transforms
    transforms = None
    if use_transforms:
        if config_path and Path(config_path).exists():
            transforms = create_overlap_transforms_from_config(config_path)
        else:
            # Use default parameters
            transforms = create_overlap_transforms_from_config(
                config_path=None, 
                radius=6.0, 
                max_neighbors=50
            )
    
    # Create dataset
    dataset = OverlapDataset(
        molecules=molecules,
        exponent_values=exponent_values,
        transform=transforms
    )
    
    return dataset

"""
Lightning DataModule for Overlap Training

This module handles data loading for the full-scale overlap training.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import lightning.pytorch as pl

from .dataset_exponent import OverlapExponentDataset, collate_overlap_data
from transforms import (
    MasterTransform,
    ConvertToTensor,
    EnforceFloat32,
    AddRadiusGraph,
    AddEdgeVectorsAndDistances,
    OverlapNormalizationTransform,
    RemoveOverlapOutliers,
)

pylogger = logging.getLogger(__name__)


class OverlapDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for overlap prediction training.
    
    Handles:
    - Loading the full molecular dataset
    - Applying train/validation/test splits
    - Creating appropriate transforms
    - Distributed data loading
    """
    
    def __init__(
        self,
        data_dir: str,
        splits_file: str,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        cache_size: int = 2000,
        use_virtual_nodes: bool = True,
        subset_size: Optional[int] = None,
        cutoff: float = 6.0,
        add_edge_vectors: bool = True,
        enforce_float32: bool = True,
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        drop_last: bool = True,
        # Normalization
        norm_dir: Optional[str] = None,
        replace_exponent: bool = True,
        replace_target: bool = True,
        cut_outliers: bool = True,
        outlier_cutoff: float = 12,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = data_dir
        self.splits_file = splits_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.cache_size = cache_size
        self.use_virtual_nodes = use_virtual_nodes
        self.subset_size = subset_size
        self.cutoff = cutoff
        self.add_edge_vectors = add_edge_vectors
        self.enforce_float32 = enforce_float32
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.drop_last = drop_last
        # Normalization settings
        self.norm_dir = norm_dir
        self.replace_exponent = replace_exponent
        self.replace_target = replace_target
        self.cut_outliers = cut_outliers
        self.outlier_cutoff = outlier_cutoff
        
        # Will be set during setup
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Metadata for model initialization
        self.metadata = {
            "num_elements": 100,  # Max atomic number supported
            "cutoff": cutoff,
            "output_size": 25,  # Fixed 25 overlap integrals per atom
        }
    
    def load_data_splits(self) -> Optional[Dict[str, List[str]]]:
        """Load train/validation/test splits from JSON file."""
        
        pylogger.info(f"Loading data splits from: {self.splits_file}")
        
        try:
            with open(self.splits_file, 'r') as f:
                splits = json.load(f)
            
            # Validate expected keys
            required_keys = ['train', 'validation', 'test']
            for key in required_keys:
                if key not in splits:
                    raise ValueError(f"Missing required key '{key}' in splits file")
            
            pylogger.info(f"Loaded splits:")
            for split, molecules in splits.items():
                pylogger.info(f"  {split}: {len(molecules):,} molecules")
            
            return splits
        
        except FileNotFoundError:
            pylogger.warning(f"Splits file not found at {self.splits_file}")
            pylogger.info("Will create default split...")
            return None
        except Exception as e:
            pylogger.error(f"Error loading splits file: {e}")
            raise
    
    def create_molecule_id_mapping(self) -> Dict[str, int]:
        """Create mapping from molecule ID to file index."""
        
        pylogger.info("Creating molecule ID mapping...")
        
        # The splits file contains integers (0, 1, 2, 3...)
        # The files are named molecule_000000.pkl, molecule_000001.pkl, etc.
        # The molecule IDs inside files are "0_0", "0_1", "0_2", etc.
        # We need to map both formats efficiently
        
        id_to_idx = {}
        
        # Create a mapping from filename to file index
        filename_to_idx = {}
        for i, filepath in enumerate(self.dataset.molecule_files):
            # Extract filename without extension (e.g., "molecule_000001" from "molecule_000001.pkl")
            filename = Path(filepath).stem
            filename_to_idx[filename] = i
        
        # Create mapping for both integer IDs (from splits) and string IDs (from molecule data)
        for i in range(len(self.dataset.molecule_files)):
            # Expected filename pattern
            expected_filename = f"molecule_{i:06d}"
            
            if expected_filename in filename_to_idx:
                file_idx = filename_to_idx[expected_filename]
                
                # Map integer ID (used in splits) to file index
                id_to_idx[i] = file_idx
                
                # Map string ID (used in molecule data) to file index  
                string_id = f"0_{i}"
                id_to_idx[string_id] = file_idx
        
        pylogger.info(f"Created efficient mapping for {len(id_to_idx):,} IDs")
        pylogger.info(f"  Integer ID range: 0 to {len(self.dataset.molecule_files)-1}")
        pylogger.info(f"  Sample mappings: 0->{id_to_idx.get(0, 'N/A')}, 10->{id_to_idx.get(10, 'N/A')}, 100->{id_to_idx.get(100, 'N/A')}")
        pylogger.info(f"  String ID samples: '0_0'->{id_to_idx.get('0_0', 'N/A')}, '0_10'->{id_to_idx.get('0_10', 'N/A')}")
        
        return id_to_idx
    
    def create_split_datasets(
        self, 
        splits: Optional[Dict[str, List[str]]]
    ) -> Tuple[Subset, Subset, Subset]:
        """Create train/validation/test dataset splits."""
        
        if splits is None or self.subset_size is not None:
            # Create default split
            pylogger.info("Creating default random split...")
            
            total_molecules = len(self.dataset.molecule_files)
            indices = torch.randperm(total_molecules)
            
            train_size = int(self.train_fraction * total_molecules)
            val_size = int(self.val_fraction * total_molecules)
            
            train_mol_indices = indices[:train_size].tolist()
            val_mol_indices = indices[train_size:train_size + val_size].tolist()
            test_mol_indices = indices[train_size + val_size:].tolist()
            
        else:
            # Use provided splits
            pylogger.info("Using provided molecule splits...")
            
            id_to_idx = self.create_molecule_id_mapping()
            
            train_mol_indices = []
            val_mol_indices = []
            test_mol_indices = []
            
            # Convert molecule IDs to indices
            for split_name, mol_ids in splits.items():
                target_list = {
                    'train': train_mol_indices,
                    'validation': val_mol_indices, 
                    'test': test_mol_indices
                }[split_name]
                
                for mol_id in mol_ids:
                    if mol_id in id_to_idx:
                        target_list.append(id_to_idx[mol_id])
                    else:
                        pylogger.warning(f"Molecule ID '{mol_id}' not found in dataset")
        
        # Convert molecule indices to data point indices using numpy for speed
        pylogger.info("Converting molecule indices to data point indices...")
        
        # Convert data_index to numpy array for vectorized operations
        import numpy as np
        data_index_array = np.array(self.dataset.data_index)
        mol_indices_array = data_index_array[:, 0]  # Extract molecule indices
        
        # Use numpy's isin for fast membership testing
        train_mask = np.isin(mol_indices_array, train_mol_indices)
        val_mask = np.isin(mol_indices_array, val_mol_indices)
        test_mask = np.isin(mol_indices_array, test_mol_indices)
        
        # Get data indices where mask is True
        train_data_indices = np.where(train_mask)[0].tolist()
        val_data_indices = np.where(val_mask)[0].tolist()
        test_data_indices = np.where(test_mask)[0].tolist()
        
        # Create subset datasets
        train_dataset = Subset(self.dataset, train_data_indices)
        val_dataset = Subset(self.dataset, val_data_indices)
        test_dataset = Subset(self.dataset, test_data_indices)
        
        pylogger.info(f"Split summary:")
        pylogger.info(f"  Train: {len(train_mol_indices):,} molecules → {len(train_dataset):,} data points")
        pylogger.info(f"  Validation: {len(val_mol_indices):,} molecules → {len(val_dataset):,} data points")
        pylogger.info(f"  Test: {len(test_mol_indices):,} molecules → {len(test_dataset):,} data points")
        
        return train_dataset, val_dataset, test_dataset
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for the given stage."""
        
        if self.dataset is None:
            # Create transforms
            transform_list = [ConvertToTensor()]
            
            if self.enforce_float32:
                transform_list.append(EnforceFloat32())

            # Apply normalization if configured (must come before graph creation)
            if self.norm_dir:
                try:
                    transform_list.append(
                        OverlapNormalizationTransform(
                            norm_dir=self.norm_dir,
                            replace_exponent=self.replace_exponent,
                            replace_target=self.replace_target,
                        )
                    )
                    pylogger.info(f"Using overlap normalization from: {self.norm_dir}")
                except Exception as e:
                    pylogger.error(f"Failed to initialize OverlapNormalizationTransform: {e}")
                    raise
            
            transform_list.append(AddRadiusGraph(radius=self.cutoff))
            
            if self.add_edge_vectors:
                transform_list.append(AddEdgeVectorsAndDistances())
            
            if self.cut_outliers:
                transform_list.append(RemoveOverlapOutliers(cutoff=self.outlier_cutoff))
            
            transforms = MasterTransform(transform_list)
            
            # Create dataset
            pylogger.info(f"Loading dataset from {self.data_dir}...")
            
            self.dataset = OverlapExponentDataset(
                data_dir=self.data_dir,
                subset_size=self.subset_size,
                use_virtual_nodes=self.use_virtual_nodes,
                cache_size=self.cache_size,
                transforms=transforms
            )
            
            pylogger.info(f"Dataset loaded: {len(self.dataset):,} data points from {len(self.dataset.molecule_files):,} molecules")
        
        if stage == "fit" or stage is None:
            if self.train_dataset is None:
                # Load splits and create datasets
                splits = self.load_data_splits()
                self.train_dataset, self.val_dataset, self.test_dataset = self.create_split_datasets(splits)
        
        if stage == "test" or stage is None:
            if self.test_dataset is None:
                # Load splits and create datasets if not already done
                splits = self.load_data_splits()
                self.train_dataset, self.val_dataset, self.test_dataset = self.create_split_datasets(splits)
    
    def train_dataloader(self):
        """Create training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_overlap_data,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=self.drop_last
        )
    
    def val_dataloader(self):
        """Create validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_overlap_data,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=False
        )
    
    def test_dataloader(self):
        """Create test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_overlap_data,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=False
        )

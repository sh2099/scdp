import json
import torch
from pathlib import Path
from torch.utils.data import Subset
from copy import deepcopy

from scdp.data.dataset import LmdbDataset
from scdp.data.datamodule import worker_init_fn
from scdp.common.pyg import DataLoader

def load_molecule_with_override(idx: int = 0, vnode: bool = False, override_atom_type: int = None, silent: bool = False):
    """
    Load a molecule and optionally override all atom types.
    
    Args:
        idx: Molecule index
        vnode: Whether to use virtual node dataset
        override_atom_type: If specified, all atoms will be set to this atomic number
        silent: If True, suppress all print output
    
    Returns:
        Modified molecule object
    """
    # Dataset paths
    if not vnode:
        data_path = "/export/scratch/mklockow/charge_density_lmdb/"
    else:
        data_path = "/export/scratch/plippman/charge_density_lmdb/"
    split_file = "/export/scratch/ialgroup/charge_density/datasplits.json"
    
    # Load dataset
    dataset = LmdbDataset(data_path)
    with open(split_file, "r") as fp:
        splits = json.load(fp)
    
    # Create subset and loader - handle both split-based and direct indexing
    if idx < len(splits['train']):
        # Use training split index mapping
        actual_idx = splits['train'][idx]
    else:
        # Direct dataset index for full dataset processing
        actual_idx = idx
    
    single_mol_dataset = Subset(dataset, [actual_idx])
    data_loader = DataLoader(
        single_mol_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        worker_init_fn=worker_init_fn
    )
    
    # Load the molecule
    molecule = next(iter(data_loader))
    
    # Get molecule ID for identification - handle list format
    raw_id = molecule.id if hasattr(molecule, 'id') and molecule.id is not None else f"idx_{idx}"
    if isinstance(raw_id, (list, tuple)) and len(raw_id) > 0:
        molecule_id = str(raw_id[0])  # Take first element and convert to string
    else:
        molecule_id = str(raw_id) if raw_id is not None else f"idx_{idx}"
    
    if not silent:
        print(f"Loading molecule {molecule_id} (dataset index {idx}) from {'vnode' if vnode else 'standard'} dataset...")
    
    # Apply atom type override if specified
    if override_atom_type is not None:
        original_types = molecule.atom_types.clone()
        original_unique = original_types.unique().tolist()
        
        # Override all atom types (including virtual nodes if present)
        molecule.atom_types = torch.full_like(molecule.atom_types, override_atom_type)
        
        if not silent:
            print(f"ATOM TYPE OVERRIDE for {molecule_id}: Changed all {len(molecule.atom_types)} atoms")
            print(f"  From types: {original_unique}")
            print(f"  To type: {override_atom_type}")
    
    # Display molecule information
    if not silent:
        print("\n" + "="*50)
        print(f"MOLECULE INFORMATION - {molecule_id}")
        print("="*50)
        print(f"Molecule ID: {molecule_id}")
        print(f"Dataset index: {idx}")
        print(f"Number of atoms: {molecule.n_atom}")
        print(f"Number of virtual nodes: {molecule.n_vnode if hasattr(molecule, 'n_vnode') else 'N/A'}")
        print(f"Total nodes: {molecule.num_nodes}")
        print(f"Number of probes: {molecule.n_probe}")
        print(f"Atom types: {molecule.atom_types.unique().tolist()}")
        print(f"Charge density min/max: {molecule.chg_labels.min():.6f} / {molecule.chg_labels.max():.6f}")
        print(f"Charge density mean/std: {molecule.chg_labels.mean():.6f} / {molecule.chg_labels.std():.6f}")
        
        if hasattr(molecule, 'is_vnode'):
            print(f"Virtual node mask available: {molecule.is_vnode.sum()} virtual nodes")
    
    return molecule

def load_molecule_direct(dataset_idx: int, vnode: bool = False, override_atom_type: int = None, silent: bool = False):
    """
    Load a molecule directly by dataset index (bypassing train/val/test splits).
    
    Args:
        dataset_idx: Direct index into the dataset
        vnode: Whether to use virtual node dataset
        override_atom_type: If specified, all atoms will be set to this atomic number
        silent: If True, suppress all print output
    
    Returns:
        Modified molecule object
    """
    # Dataset paths
    if not vnode:
        data_path = "/export/scratch/mklockow/charge_density_lmdb/"
    else:
        data_path = "/export/scratch/plippman/charge_density_lmdb/"
    
    # Load dataset
    dataset = LmdbDataset(data_path)
    
    # Create subset and loader using direct index
    single_mol_dataset = Subset(dataset, [dataset_idx])
    data_loader = DataLoader(
        single_mol_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        worker_init_fn=worker_init_fn
    )
    
    # Load the molecule
    molecule = next(iter(data_loader))
    
    # Get molecule ID for identification - handle list format
    raw_id = molecule.id if hasattr(molecule, 'id') and molecule.id is not None else f"idx_{dataset_idx}"
    if isinstance(raw_id, (list, tuple)) and len(raw_id) > 0:
        molecule_id = str(raw_id[0])
    else:
        molecule_id = str(raw_id) if raw_id is not None else f"idx_{dataset_idx}"
    
    if not silent:
        print(f"Loading molecule {molecule_id} (dataset index {dataset_idx}) from {'vnode' if vnode else 'standard'} dataset...")
    
    # Apply atom type override if specified
    if override_atom_type is not None:
        original_types = molecule.atom_types.clone()
        original_unique = original_types.unique().tolist()
        molecule.atom_types = torch.full_like(molecule.atom_types, override_atom_type)
        
        if not silent:
            print(f"ATOM TYPE OVERRIDE for {molecule_id}: Changed all {len(molecule.atom_types)} atoms")
            print(f"  From types: {original_unique}")
            print(f"  To type: {override_atom_type}")
    
    # Display molecule information
    if not silent:
        print("\n" + "="*50)
        print(f"MOLECULE INFORMATION - {molecule_id}")
        print("="*50)
        print(f"Molecule ID: {molecule_id}")
        print(f"Dataset index: {dataset_idx}")
        print(f"Number of atoms: {molecule.n_atom}")
        print(f"Number of virtual nodes: {molecule.n_vnode if hasattr(molecule, 'n_vnode') else 'N/A'}")
        print(f"Total nodes: {molecule.num_nodes}")
        print(f"Number of probes: {molecule.n_probe}")
        print(f"Atom types: {molecule.atom_types.unique().tolist()}")
        print(f"Charge density min/max: {molecule.chg_labels.min():.6f} / {molecule.chg_labels.max():.6f}")
        print(f"Charge density mean/std: {molecule.chg_labels.mean():.6f} / {molecule.chg_labels.std():.6f}")
        
        if hasattr(molecule, 'is_vnode'):
            print(f"Virtual node mask available: {molecule.is_vnode.sum()} virtual nodes")
    
    return molecule

def get_atom_centers_and_types(molecule, use_vnodes: bool = False, override_atom_type: int = None, silent: bool = False):
    """
    Extract atom centers and types for basis function placement.
    
    Args:
        molecule: Loaded molecule object
        use_vnodes: Whether to include virtual nodes
        override_atom_type: Override all atom types to this value
        silent: If True, suppress all print output
    
    Returns:
        Tuple of (atom_coords, atom_types, info_string)
    """
    # Handle molecule ID formatting - handle list format
    raw_id = molecule.id if hasattr(molecule, 'id') and molecule.id is not None else "UNKNOWN"
    if isinstance(raw_id, (list, tuple)) and len(raw_id) > 0:
        molecule_id = str(raw_id[0])  # Take first element and convert to string
    else:
        molecule_id = str(raw_id) if raw_id is not None else "UNKNOWN"
    
    if use_vnodes and hasattr(molecule, 'is_vnode'):
        # Use all atoms and virtual nodes
        atom_coords = molecule.coords
        atom_types = molecule.atom_types
        
        real_mask = ~molecule.is_vnode
        vnode_mask = molecule.is_vnode
        
        info = f"Molecule {molecule_id}: {len(atom_coords)} centers ({real_mask.sum().item()} real atoms + {vnode_mask.sum().item()} virtual nodes)"
        if not silent:
            print(f"Using {len(atom_coords)} centers for {molecule_id}:")
            print(f"  Real atoms: {real_mask.sum().item()} (types: {atom_types[real_mask].unique().tolist()})")
            print(f"  Virtual nodes: {vnode_mask.sum().item()} (types: {atom_types[vnode_mask].unique().tolist()})")
        
    else:
        # Use only real atoms (non-zero atom types)
        real_mask = molecule.atom_types != 0
        atom_coords = molecule.coords[real_mask]
        atom_types = molecule.atom_types[real_mask]
        
        info = f"Molecule {molecule_id}: {len(atom_coords)} real atoms (types: {atom_types.unique().tolist()})"
        if not silent:
            print(f"Using {len(atom_coords)} real atoms for {molecule_id} (types: {atom_types.unique().tolist()})")
    
    # Apply override after filtering if specified
    if override_atom_type is not None:
        original_unique = atom_types.unique().tolist()
        atom_types = torch.full_like(atom_types, override_atom_type)
        if not silent:
            print(f"Applied override for {molecule_id}: all types changed to {override_atom_type}")
        info += f" → override to type {override_atom_type}"
    
    return atom_coords, atom_types, info

def get_dataset_size(vnode: bool = False) -> int:
    """Get the total size of the dataset."""
    if not vnode:
        data_path = "/export/scratch/mklockow/charge_density_lmdb/"
    else:
        data_path = "/export/scratch/plippman/charge_density_lmdb/"
    
    dataset = LmdbDataset(data_path)
    return len(dataset)

if __name__ == "__main__":
    # Test the new loader
    molecule = load_molecule_with_override(idx=36729, vnode=True, override_atom_type=6)
    print(molecule.id)
    coords, types, info = get_atom_centers_and_types(molecule, use_vnodes=True)
    print(f"\nExtracted {len(coords)} centers with types {types.unique().tolist()}")

if __name__ == "__main__":
    # Test the new loader
    molecule = load_molecule_with_override(idx=36729, vnode=True, override_atom_type=6)
    print(molecule.id)
    coords, types, info = get_atom_centers_and_types(molecule, use_vnodes=True)
    print(f"\nExtracted {len(coords)} centers with types {types.unique().tolist()}")

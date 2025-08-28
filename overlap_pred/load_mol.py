import json
import torch
from pathlib import Path
from torch.utils.data import Subset
from copy import deepcopy

from scdp.data.dataset import LmdbDataset
from scdp.data.datamodule import worker_init_fn
from scdp.common.pyg import DataLoader

def load_molecule_with_override(idx: int = 0, vnode: bool = False, override_atom_type: int = None):
    """
    Load a molecule and optionally override all atom types.
    
    Args:
        idx: Molecule index
        vnode: Whether to use virtual node dataset
        override_atom_type: If specified, all atoms will be set to this atomic number
    
    Returns:
        Modified molecule object
    """
    # Dataset paths
    if not vnode:
        data_path = "/export/scratch/mklockow/charge_density_lmdb/"
    else:
        data_path = "/export/scratch/plippman/charge_density_lmdb/"
    split_file = "/export/scratch/ialgroup/charge_density/datasplits.json"
    
    print(f"Loading molecule {idx} from {'vnode' if vnode else 'standard'} dataset...")
    
    # Load dataset
    dataset = LmdbDataset(data_path)
    with open(split_file, "r") as fp:
        splits = json.load(fp)
    
    # Create subset and loader
    single_mol_dataset = Subset(dataset, [splits['train'][idx]])
    data_loader = DataLoader(
        single_mol_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        worker_init_fn=worker_init_fn
    )
    
    # Load the molecule
    molecule = next(iter(data_loader))
    
    # Apply atom type override if specified
    if override_atom_type is not None:
        original_types = molecule.atom_types.clone()
        original_unique = original_types.unique().tolist()
        
        # Override all atom types (including virtual nodes if present)
        molecule.atom_types = torch.full_like(molecule.atom_types, override_atom_type)
        
        print(f"ATOM TYPE OVERRIDE: Changed all {len(molecule.atom_types)} atoms")
        print(f"  From types: {original_unique}")
        print(f"  To type: {override_atom_type}")
    
    # Display molecule information
    print("\n" + "="*50)
    print("MOLECULE INFORMATION")
    print("="*50)
    print(f"Molecule index: {idx}")
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

def get_atom_centers_and_types(molecule, use_vnodes: bool = False, override_atom_type: int = None):
    """
    Extract atom centers and types for basis function placement.
    
    Args:
        molecule: Loaded molecule object
        use_vnodes: Whether to include virtual nodes
        override_atom_type: Override all atom types to this value
    
    Returns:
        Tuple of (atom_coords, atom_types, info_dict)
    """
    if use_vnodes and hasattr(molecule, 'is_vnode'):
        # Use all atoms and virtual nodes
        atom_coords = molecule.coords
        atom_types = molecule.atom_types
        
        real_mask = ~molecule.is_vnode
        vnode_mask = molecule.is_vnode
        
        info = {
            'total_centers': len(atom_coords),
            'real_atoms': real_mask.sum().item(),
            'virtual_nodes': vnode_mask.sum().item(),
            'real_atom_types': atom_types[real_mask].unique().tolist(),
            'vnode_types': atom_types[vnode_mask].unique().tolist()
        }
        
        print(f"Using {info['total_centers']} centers:")
        print(f"  Real atoms: {info['real_atoms']} (types: {info['real_atom_types']})")
        print(f"  Virtual nodes: {info['virtual_nodes']} (types: {info['vnode_types']})")
        
    else:
        # Use only real atoms (non-zero atom types)
        real_mask = molecule.atom_types != 0
        atom_coords = molecule.coords[real_mask]
        atom_types = molecule.atom_types[real_mask]
        
        info = {
            'total_centers': len(atom_coords),
            'real_atoms': len(atom_coords),
            'virtual_nodes': 0,
            'real_atom_types': atom_types.unique().tolist(),
            'vnode_types': []
        }
        
        print(f"Using {info['total_centers']} real atoms (types: {info['real_atom_types']})")
    
    # Apply override after filtering if specified
    if override_atom_type is not None:
        original_unique = atom_types.unique().tolist()
        atom_types = torch.full_like(atom_types, override_atom_type)
        print(f"Applied override: all types changed to {override_atom_type}")
        info['override_applied'] = True
        info['original_types'] = original_unique
    else:
        info['override_applied'] = False
    
    return atom_coords, atom_types, info

if __name__ == "__main__":
    # Test the new loader
    molecule = load_molecule_with_override(idx=5, vnode=True, override_atom_type=6)
    coords, types, info = get_atom_centers_and_types(molecule, use_vnodes=True)
    print(f"\nExtracted {len(coords)} centers with types {types.unique().tolist()}")

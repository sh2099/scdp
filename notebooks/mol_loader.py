import json
import torch
from pathlib import Path
from torch.utils.data import Subset

from scdp.data.dataset import LmdbDataset
from scdp.data.datamodule import worker_init_fn
from scdp.common.pyg import DataLoader

def load_single_molecule(idx: int = 0, vnode=False):
    """Load and examine a single molecule from the dataset."""
    
    # Dataset paths
    if not vnode:
        data_path = "/export/scratch/mklockow/charge_density_lmdb/"
    else:
        data_path = "/export/scratch/plippman/charge_density_lmdb/"
    split_file = "/export/scratch/ialgroup/charge_density/datasplits.json"
    print("Loading dataset...")
    # Use LmdbDataset object
    dataset = LmdbDataset(data_path)
    print(f"Total dataset size: {len(dataset)}")
    
    # Load splits
    with open(split_file, "r") as fp:
        splits = json.load(fp)
    
    print(f"Train samples: {len(splits['train'])}")
    print(f"Validation samples: {len(splits['validation'])}")
    print(f"Test samples: {len(splits['test'])}")
    
    # Create a subset with just the first molecule from the training set
    single_mol_dataset = Subset(dataset, [splits['train'][idx]])
    
    # Create data loader
    data_loader = DataLoader(
        single_mol_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  
        worker_init_fn=worker_init_fn
    )
    
    # Load the molecule
    print("\nLoading molecule...")
    molecule = next(iter(data_loader))
    
    # Display molecule information
    print("\n" + "="*50)
    print("MOLECULE INFORMATION")
    print("="*50)
    
    print(f"Metadata: {molecule.metadata}")
    print(f"Build method: {molecule.build_method}")
    print(f"Number of atoms: {molecule.n_atom}")
    print(f"Number of virtual nodes: {molecule.n_vnode if hasattr(molecule, 'n_vnode') else 'N/A'}")
    print(f"Total nodes: {molecule.num_nodes}")
    print(f"Number of probes: {molecule.n_probe}")
    
    print(f"\nAtom types: {molecule.atom_types}")
    print(f"Atom coordinates shape: {molecule.coords.shape}")
    print(f"Cell matrix shape: {molecule.cell.shape}")
    print(f"Edge index shape: {molecule.edge_index.shape}")
    print(f"Number of edges: {molecule.edge_index.shape[1]}")
    
    print(f"\nProbe coordinates shape: {molecule.probe_coords.shape}")
    print(f"Charge density labels shape: {molecule.chg_labels.shape}")
    print(f"Charge density min/max: {molecule.chg_labels.min():.6f} / {molecule.chg_labels.max():.6f}")
    print(f"Charge density mean/std: {molecule.chg_labels.mean():.6f} / {molecule.chg_labels.std():.6f}")
    
    if hasattr(molecule, 'grid_size'):
        print(f"Grid size: {molecule.grid_size}")
    
    if hasattr(molecule, 'vnode_method'):
        print(f"Virtual node method: {molecule.vnode_method}")
    
    if hasattr(molecule, 'is_vnode'):
        print(f"Virtual node mask shape: {molecule.is_vnode.shape}")
        print(f"Number of virtual nodes (from mask): {molecule.is_vnode.sum()}")
    
    # Show some coordinate examples
    print(f"\nFirst 3 atom coordinates:")
    print(molecule.coords[:3])
    
    print(f"\nFirst 3 probe coordinates:")
    print(molecule.probe_coords[:3])
    
    print(f"\nFirst 5 charge density values:")
    print(molecule.chg_labels[:5])
    
    # Sample some probes if this is a vnode-based molecule
    if hasattr(molecule, 'sample_probe'):
        print(f"\n" + "="*50)
        print("SAMPLING PROBES")
        print("="*50)
        
        sampled_mol = molecule.sample_probe(n_probe=100)
        print(f"Sampled molecule:")
        print(f"  Number of probes: {sampled_mol.n_probe}")
        print(f"  Probe coordinates shape: {sampled_mol.probe_coords.shape}")
        print(f"  Charge labels shape: {sampled_mol.chg_labels.shape}")
        print(f"  Sampled flag: {sampled_mol.sampled}")
    
    return molecule

if __name__ == "__main__":
    molecule = load_single_molecule(10)
    print(f"\nMolecule loaded successfully!")
    print(f"Available attributes: {[attr for attr in dir(molecule) if not attr.startswith('_')]}")

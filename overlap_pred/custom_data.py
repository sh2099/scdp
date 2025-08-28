from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict, Iterable
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


def _first_attr(obj, names: Iterable[str]):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


@dataclass
class CustomMolecule:
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
    n_probe: Optional[int] = None
    n_vnode: Optional[int] = None
    
    # method information
    build_method: Optional[str] = None
    vnode_method: Optional[str] = None
    
    # overlap data - 2D structure (exponents x basis functions)
    overlap_int_2d: Optional[torch.Tensor] = None
    overlap_mapping_info: Optional[Dict[str, Any]] = None

    def __repr__(self):
        overlap_shape = self.overlap_int_2d.shape if self.overlap_int_2d is not None else None
        # Format ID properly if it's a list
        display_id = self.id
        if isinstance(self.id, (list, tuple)) and len(self.id) > 0:
            display_id = str(self.id[0])
        elif self.id is not None:
            display_id = str(self.id)
        
        return (
            f"CustomMolecule(n_atom={self.n_atom}, n_probe={self.n_probe}, "
            f"n_vnode={self.n_vnode}, overlap_int_shape={overlap_shape}, "
            f"id={display_id}, build_method={self.build_method})"
        )

    @property
    def overlap_integrals_1d(self):
        """Compute 1D overlap integrals by summing over exponents for backward compatibility."""
        if self.overlap_int_2d is not None:
            return self.overlap_int_2d.sum(dim=0)
        return None

    @classmethod
    def from_scdp_data(cls, data_obj, overlap_int_2d: Optional[torch.Tensor] = None, 
                      overlap_mapping_info: Optional[Dict] = None, keep_probe_grid: bool = False):
        """
        Construct CustomMolecule from a scdp Data-like object.
        - copies all specified fields and converts to torch tensors when appropriate
        - omits heavy grid-related fields by default (chg_labels, probe_coords, etc.)
        - optionally keeps them if keep_probe_grid=True
        - overlap_int_2d should be (n_exponents, n_basis_functions) shape
        - overlap_mapping_info contains exponent values and basis function mapping
        """
        # Extract core molecular properties
        atom_types = _first_attr(data_obj, ["atom_types", "z", "atomic_numbers"])
        coords = _first_attr(data_obj, ["coords", "atom_coords", "pos", "atom_pos"])
        batch = _first_attr(data_obj, ["batch"])
        cell = _first_attr(data_obj, ["cell"])
        is_vnode = _first_attr(data_obj, ["is_vnode"])
        node_attrs = _first_attr(data_obj, ["node_attrs"])
        ptr = _first_attr(data_obj, ["ptr"])
        
        # Extract identifiers and counts
        mol_id = _first_attr(data_obj, ["id", "mol_id", "molecule_id"])
        
        # Format molecule ID properly if it's a list
        if isinstance(mol_id, (list, tuple)) and len(mol_id) > 0:
            mol_id = str(mol_id[0])  # Take first element and convert to string
        elif mol_id is not None:
            mol_id = str(mol_id)
        
        # Extract counts
        n_atom = _first_attr(data_obj, ["n_atom", "natoms", "num_atoms"])
        n_probe = _first_attr(data_obj, ["n_probe", "n_probes", "num_probes"])
        n_vnode = _first_attr(data_obj, ["n_vnode", "num_vnodes"])
        
        # Extract method information
        build_method = _first_attr(data_obj, ["build_method"])
        vnode_method = _first_attr(data_obj, ["vnode_method"])
        
        # Convert tensors
        atom_types_t = _to_tensor(atom_types) if atom_types is not None else None
        coords_t = _to_tensor(coords) if coords is not None else None
        batch_t = _to_tensor(batch) if batch is not None else None
        cell_t = _to_tensor(cell) if cell is not None else None
        is_vnode_t = _to_tensor(is_vnode) if is_vnode is not None else None
        node_attrs_t = _to_tensor(node_attrs) if node_attrs is not None else None
        ptr_t = _to_tensor(ptr) if ptr is not None else None

        # build metadata: copy small items, avoid large arrays
        metadata = {}
        for key in ["name", "smiles", "formula", "charge", "energy", "forces"]:
            if hasattr(data_obj, key):
                val = getattr(data_obj, key)
                # avoid storing large arrays in metadata
                if not isinstance(val, (torch.Tensor, np.ndarray)) or (hasattr(val, 'numel') and val.numel() < 100):
                    metadata[key] = val

        # include a light fingerprint of atom_types unique values
        try:
            if atom_types_t is not None:
                metadata["unique_atom_types"] = torch.unique(atom_types_t).cpu().tolist()
        except Exception:
            pass

        inst = cls(
            atom_types=atom_types_t,
            coords=coords_t,
            batch=batch_t,
            cell=cell_t,
            is_vnode=is_vnode_t,
            node_attrs=node_attrs_t,
            ptr=ptr_t,
            id=mol_id,
            metadata=metadata,
            n_atom=int(n_atom) if n_atom is not None else None,
            n_probe=int(n_probe) if n_probe is not None else None,
            n_vnode=int(n_vnode) if n_vnode is not None else None,
            build_method=str(build_method) if build_method is not None else None,
            vnode_method=str(vnode_method) if vnode_method is not None else None,
            overlap_int_2d=_to_tensor(overlap_int_2d) if overlap_int_2d is not None else None,
            overlap_mapping_info=overlap_mapping_info,
        )

        # Optionally attach heavy grid items (only if explicitly requested).
        if keep_probe_grid:
            for name in ["chg_labels", "probe_coords", "weights", "probe_weights", "ao_values", "ao_coeffs"]:
                if hasattr(data_obj, name):
                    setattr(inst, name, getattr(data_obj, name))

        return inst

    def save_pickle(self, path):
        """Save the CustomMolecule object as a pickle file."""
        path = Path(path)
        with open(path, "wb") as fp:
            pickle.dump(self, fp)

    @classmethod
    def load_pickle(cls, path):
        """Load a CustomMolecule object from a pickle file."""
        path = Path(path)
        with open(path, "rb") as fp:
            return pickle.load(fp)

    def to_dict(self):
        """Return a JSON-serializable dict (tensors -> lists)."""
        d = {}
        for k, v in asdict(self).items():
            if k == "metadata":
                d[k] = self.metadata
                continue
            elif k == "overlap_mapping_info":
                # Handle mapping info specially - convert tensor values to lists
                if v is not None:
                    mapping_dict = {}
                    for key, val in v.items():
                        if key == "mapping":
                            # Convert list of tuples to serializable format
                            mapping_dict[key] = val  # Already serializable
                        elif isinstance(val, dict):
                            mapping_dict[key] = {k2: v2 for k2, v2 in val.items()}
                        else:
                            mapping_dict[key] = val
                    d[k] = mapping_dict
                else:
                    d[k] = None
                continue
            
            arr = _to_numpy(v)
            if arr is None:
                d[k] = None
            else:
                d[k] = arr.tolist() if isinstance(arr, np.ndarray) else arr
        return d

    def save_json(self, path):
        with open(path, "w") as fp:
            json.dump(self.to_dict(), fp)

    @classmethod
    def load_json(cls, path):
        with open(path, "r") as fp:
            d = json.load(fp)
        # rehydrate into tensors where appropriate
        atom_types = torch.as_tensor(d.get("atom_types")) if d.get("atom_types") is not None else None
        coords = torch.as_tensor(d.get("coords")) if d.get("coords") is not None else None
        batch = torch.as_tensor(d.get("batch")) if d.get("batch") is not None else None
        cell = torch.as_tensor(d.get("cell")) if d.get("cell") is not None else None
        is_vnode = torch.as_tensor(d.get("is_vnode")) if d.get("is_vnode") is not None else None
        node_attrs = torch.as_tensor(d.get("node_attrs")) if d.get("node_attrs") is not None else None
        ptr = torch.as_tensor(d.get("ptr")) if d.get("ptr") is not None else None
        overlap_int_2d = torch.as_tensor(d.get("overlap_int_2d")) if d.get("overlap_int_2d") is not None else None
        
        return cls(
            atom_types=atom_types,
            coords=coords,
            batch=batch,
            cell=cell,
            is_vnode=is_vnode,
            node_attrs=node_attrs,
            ptr=ptr,
            id=d.get("id"),
            metadata=d.get("metadata", {}),
            n_atom=d.get("n_atom"),
            n_probe=d.get("n_probe"),
            n_vnode=d.get("n_vnode"),
            build_method=d.get("build_method"),
            vnode_method=d.get("vnode_method"),
            overlap_int_2d=overlap_int_2d,
            overlap_mapping_info=d.get("overlap_mapping_info"),
        )

    def get_overlap_for_exponent(self, exponent_idx: int):
        """Get overlap values for a specific exponent index."""
        if self.overlap_int_2d is not None:
            return self.overlap_int_2d[exponent_idx, :]
        return None

    def get_overlap_for_atom_L_m(self, atom_idx: int, L: int, m: int):
        """Get overlap values for specific atom, L, m quantum numbers across all exponents."""
        if self.overlap_mapping_info is None or self.overlap_int_2d is None:
            return None
        
        atom_basis_structure = self.overlap_mapping_info.get('atom_basis_structure', {})
        if atom_idx not in atom_basis_structure:
            return None
        
        atom_struct = atom_basis_structure[atom_idx]
        if L not in atom_struct or m not in atom_struct[L]:
            return None
        
        # Get basis indices for this (atom, L, m) combination
        basis_indices = [item['basis_idx'] for item in atom_struct[L][m]]
        
        if basis_indices:
            return self.overlap_int_2d[:, basis_indices]
        return None

    def analyze_overlap_by_L(self):
        """Analyze overlap contributions by L value for each atom."""
        if self.overlap_mapping_info is None or self.overlap_int_2d is None:
            return None
        
        atom_basis_structure = self.overlap_mapping_info.get('atom_basis_structure', {})
        analysis = {}
        
        for atom_idx, atom_struct in atom_basis_structure.items():
            analysis[atom_idx] = {}
            for L in atom_struct.keys():
                # Collect all basis indices for this L value
                L_basis_indices = []
                for m in atom_struct[L]:
                    for item in atom_struct[L][m]:
                        L_basis_indices.append(item['basis_idx'])
                
                if L_basis_indices:
                    L_overlaps = self.overlap_int_2d[:, L_basis_indices]
                    analysis[atom_idx][L] = {
                        'total_overlap': L_overlaps.sum().item(),
                        'mean_overlap': L_overlaps.mean().item(),
                        'max_overlap': L_overlaps.max().item(),
                        'n_basis_functions': len(L_basis_indices)
                    }
        
        return analysis

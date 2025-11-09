"""
Transforms for overlap prediction model to generate graph structure compatible with ESCN.

These transforms convert CustomMolecule data into the graph format expected by ESCN,
including edge indices, edge vectors, distances, and other properties needed for
message passing in the GNN.
"""

import torch
import numpy as np
from typing import Optional, Union, List
from pathlib import Path
import pickle
from torch_geometric.nn import radius_graph

# Spans for L=0..4 within per-atom basis (start inclusive, end exclusive)
L_SPANS = [(0, 1), (1, 4), (4, 9), (9, 16), (16, 25)]

# For consistency with your transform demo structure
class MasterTransform:
    """Apply a sequence of transforms to data."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data_object):
        for transform in self.transforms:
            data_object = transform(data_object)
        return data_object


class ConvertToTensor:
    """Convert relevant fields to tensors."""
    
    def __init__(self):
        pass
    
    def __call__(self, data: dict) -> dict:
        # Convert coordinates and atom types to tensors if they aren't already
        if isinstance(data['coords'], np.ndarray):
            data['coords'] = torch.from_numpy(data['coords']).float()
        if isinstance(data['atom_types'], np.ndarray):
            data['atom_types'] = torch.from_numpy(data['atom_types']).long()
        
        # Convert exponent value to tensor
        if isinstance(data['exponent_value'], (int, float, np.number)):
            data['exponent_value'] = torch.tensor([data['exponent_value']], dtype=torch.float32)
        elif isinstance(data['exponent_value'], np.ndarray):
            data['exponent_value'] = torch.from_numpy(data['exponent_value']).float()
        
        # Convert overlap target to tensor if needed
        if isinstance(data['target'], np.ndarray):
            data['target'] = torch.from_numpy(data['target']).float()
        
        return data


class EnforceFloat32:
    """
    Enforce float32 precision for all floating-point tensors to prevent dtype mismatches.
    
    This transform ensures all float tensors are in float32 format, which is consistent
    with most PyTorch models and prevents common 'Found dtype Double but expected Float' errors.
    """
    
    def __init__(self):
        pass
    
    def __call__(self, data: dict) -> dict:
        """Convert all float tensors to float32 precision."""
        for key, value in data.items():
            if torch.is_tensor(value):
                # Convert floating-point tensors to float32
                if value.dtype in [torch.float16, torch.float32, torch.float64]:
                    data[key] = value.to(dtype=torch.float32)
                # Leave integer and boolean tensors unchanged
            elif isinstance(value, list) and len(value) > 0 and torch.is_tensor(value[0]):
                # Handle lists of tensors (like targets)
                if value[0].dtype in [torch.float16, torch.float32, torch.float64]:
                    data[key] = [t.to(dtype=torch.float32) for t in value]
        
        return data


class AddRadiusGraph:
    """
    Add graph connectivity based on atomic distances within a cutoff radius.
    
    This creates the edge_index and shifts needed by ESCN, similar to how
    radius_graph_pbc works in the original data processing.
    """
    
    def __init__(
        self, 
        radius: float = 6.0, 
        max_neighbors: int = 50,
        include_self_edges: bool = False,
    ):
        """
        Args:
            radius: Cutoff distance for neighbor finding
            max_neighbors: Maximum number of neighbors per atom
            include_self_edges: Whether to include self-connections
        """
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.include_self_edges = include_self_edges

    def __call__(self, data):
        """
        Add edge_index and shifts to the data.
        
        Expects data to have:
        - 'coords': atomic coordinates [n_atoms, 3]
        - 'atom_types': atomic numbers [n_atoms]
        
        Adds:
        - 'edge_index': connectivity [2, n_edges]
        - 'shifts': PBC shifts [n_edges, 3] (zero for non-periodic)
        """
        coords = data['coords']
        n_atoms = len(coords)
        
        # Use PyTorch Geometric's radius_graph function
        # This is much more reliable than our custom implementation
        edge_index = radius_graph(
            coords, 
            r=self.radius, 
            max_num_neighbors=self.max_neighbors if self.max_neighbors > 0 else None,
            loop=self.include_self_edges  # loop parameter controls self-edges
        )
        
        # Set data properties
        data['edge_index'] = edge_index
        
        # Add shifts for PBC (set to zero for non-periodic systems)
        if len(edge_index[0]) > 0:
            data['shifts'] = torch.zeros(len(edge_index[0]), 3, dtype=coords.dtype, device=coords.device)
        else:
            data['shifts'] = torch.empty((0, 3), dtype=coords.dtype, device=coords.device)
        
        return data


class AddEdgeVectorsAndDistances:
    """
    Compute edge vectors and distances from coordinates and edge indices.
    
    This mirrors the get_edge_vectors_and_lengths function used in ESCN.
    """
    
    def __init__(self, normalize_vectors: bool = False, eps: float = 1e-9):
        """
        Args:
            normalize_vectors: Whether to normalize edge vectors
            eps: Small value to avoid division by zero
        """
        self.normalize_vectors = normalize_vectors
        self.eps = eps

    def __call__(self, data):
        """
        Add edge vectors and distances.
        
        Expects data to have:
        - 'coords': atomic coordinates [n_atoms, 3]
        - 'edge_index': connectivity [2, n_edges]
        - 'shifts': PBC shifts [n_edges, 3]
        
        Adds:
        - 'edge_distance_vec': edge vectors [n_edges, 3]
        - 'edge_distance': edge lengths [n_edges, 1]
        """
        coords = data['coords']
        edge_index = data['edge_index']
        shifts = data['shifts']
        
        if len(edge_index[0]) == 0:
            # No edges
            data['edge_distance_vec'] = torch.empty((0, 3), dtype=coords.dtype, device=coords.device)
            data['edge_distance'] = torch.empty((0, 1), dtype=coords.dtype, device=coords.device)
            return data
        
        # Compute edge vectors: target - source + shift
        sender = edge_index[0]
        receiver = edge_index[1]
        vectors = coords[receiver] - coords[sender] + shifts  # [n_edges, 3]
        
        # Compute edge lengths
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
        
        if self.normalize_vectors:
            vectors_normed = vectors / (lengths + self.eps)
            data['edge_distance_vec'] = vectors_normed
        else:
            data['edge_distance_vec'] = vectors
            
        data['edge_distance'] = lengths
        
        return data


class AddBatchIndices:
    """
    Add batch indices for batching multiple molecules together.
    
    For single molecules, all atoms belong to batch 0.
    For multiple molecules, this would need to be set appropriately.
    """
    
    def __init__(self, batch_idx: int = 0):
        """
        Args:
            batch_idx: Batch index for this molecule
        """
        self.batch_idx = batch_idx

    def __call__(self, data):
        """
        Add batch indices.
        
        Expects data to have:
        - 'atom_types': atomic numbers [n_atoms]
        
        Adds:
        - 'batch': batch indices [n_atoms]
        """
        n_atoms = len(data['atom_types'])
        data['batch'] = torch.full((n_atoms,), self.batch_idx, dtype=torch.long)
        return data


class ConvertToTensor:
    """
    Convert numpy arrays and other data types to PyTorch tensors.
    
    Similar to the ToTorch transform in your demo.
    """
    
    def __init__(
        self, 
        device: Optional[torch.device] = None,
        float_dtype: torch.dtype = torch.float32,
        int_dtype: torch.dtype = torch.long
    ):
        """
        Args:
            device: Target device for tensors
            float_dtype: Data type for floating point tensors
            int_dtype: Data type for integer tensors
        """
        self.device = device
        self.float_dtype = float_dtype
        self.int_dtype = int_dtype

    def __call__(self, data):
        """
        Convert arrays to tensors.
        
        Converts:
        - coords, exponent_value -> float_dtype
        - atom_types, edge_index, batch -> int_dtype
        - other arrays -> appropriate dtype
        """
        # List of keys that should be float tensors
        float_keys = ['coords', 'exponent_value', 'shifts', 'edge_distance_vec', 'edge_distance']
        
        # List of keys that should be int tensors  
        int_keys = ['atom_types', 'edge_index', 'batch', 'exponent_index']
        
        for key, value in data.items():
            if key in float_keys and not isinstance(value, torch.Tensor):
                data[key] = torch.tensor(value, dtype=self.float_dtype, device=self.device)
            elif key in int_keys and not isinstance(value, torch.Tensor):
                data[key] = torch.tensor(value, dtype=self.int_dtype, device=self.device)
            elif isinstance(value, (np.ndarray, list)) and not isinstance(value, torch.Tensor):
                # Auto-detect dtype for other arrays
                if isinstance(value, np.ndarray):
                    if value.dtype.kind in ['i', 'u']:  # integer types
                        data[key] = torch.tensor(value, dtype=self.int_dtype, device=self.device)
                    else:  # float types
                        data[key] = torch.tensor(value, dtype=self.float_dtype, device=self.device)
                else:
                    data[key] = torch.tensor(value, device=self.device)
            elif isinstance(value, torch.Tensor) and self.device is not None:
                data[key] = value.to(self.device)
        
        return data


class OverlapNormalizationTransform:
    """
    Normalize exponent and overlap targets using precomputed curves.

    - Exponent: x = log(a/alpha_min) / log(alpha_max/alpha_min), clipped to [0,1].
    - Overlap (per L): z = (O - mu_L(x)) / sigma_L(x), where mu/std are read from saved curves.

    Expects caches produced by scripts/fit_overlap_normalization.py:
      - overlap_norm_meta.json: contains alpha_min, alpha_max
      - overlap_fit_curves.npz: contains x_grid, mean_grid (5xN), std_grid (5xN)
    """

    def __init__(
        self,
        norm_dir: Union[str, Path],
        *,
        replace_exponent: bool = True,
        replace_target: bool = True,
        store_original: bool = True,
        std_clip: float = 1e-12,
    ) -> None:
        self.norm_dir = Path(norm_dir)
        self.replace_exponent = replace_exponent
        self.replace_target = replace_target
        self.store_original = store_original
        self.std_clip = float(std_clip)

        meta_path = self.norm_dir / "overlap_norm_meta.json"
        curves_path = self.norm_dir / "overlap_fit_curves.npz"
        if not meta_path.exists() or not curves_path.exists():
            raise FileNotFoundError(
                f"Normalization cache files not found in {self.norm_dir}. "
                f"Expected 'overlap_norm_meta.json' and 'overlap_fit_curves.npz'."
            )
        # Load min/max
        import json as _json
        with open(meta_path, "r") as f:
            meta = _json.load(f)
        self.alpha_min = float(meta["alpha_min"]) if "alpha_min" in meta else None
        self.alpha_max = float(meta["alpha_max"]) if "alpha_max" in meta else None
        if not (np.isfinite(self.alpha_min) and np.isfinite(self.alpha_max) and self.alpha_min > 0 and self.alpha_max > 0):
            raise ValueError("Invalid alpha_min/alpha_max in normalization meta")

        # Load fitted curves (grids)
        z = np.load(curves_path)
        self.x_grid: np.ndarray = z["x_grid"].astype(np.float64)
        self.mean_grid: np.ndarray = z["mean_grid"].astype(np.float64)  # shape (5, N)
        self.std_grid: np.ndarray = z["std_grid"].astype(np.float64)    # shape (5, N)
        if self.mean_grid.shape[0] < 5 or self.std_grid.shape[0] < 5:
            raise ValueError("Expected 5 L-channels in fitted curves")
        # Ensure grids are sorted
        if not (np.all(np.diff(self.x_grid) >= 0)):
            order = np.argsort(self.x_grid)
            self.x_grid = self.x_grid[order]
            self.mean_grid = self.mean_grid[:, order]
            self.std_grid = self.std_grid[:, order]

    def _log_normalize_exp(self, a: float) -> float:
        a = float(a)
        amin = float(self.alpha_min)
        amax = float(self.alpha_max)
        if a <= 0 or not (np.isfinite(amin) and np.isfinite(amax)):
            return 0.0
        if np.isclose(amin, amax):
            return 0.0
        denom = np.log(amax / amin)
        if denom == 0.0:
            return 0.0
        a_clamped = min(max(a, amin), amax)
        x = np.log(a_clamped / amin) / denom
        return float(np.clip(x, 0.0, 1.0))

    def _interp_1d(self, ygrid: np.ndarray, x: float) -> float:
        # Linear interpolation on pre-sampled grid
        return float(np.interp(x, self.x_grid, ygrid))

    def __call__(self, data: dict) -> dict:
        # Handle exponent normalization
        exp_val = data.get("exponent_value")
        if exp_val is None:
            return data
        # Convert to Python float
        if isinstance(exp_val, torch.Tensor):
            a = float(exp_val.detach().cpu().view(-1)[0].item())
        else:
            a = float(exp_val if not isinstance(exp_val, (list, tuple, np.ndarray)) else np.asarray(exp_val).reshape(-1)[0])

        x = self._log_normalize_exp(a)

        if self.store_original:
            data['exponent_value_raw'] = exp_val
        # Store normalized exponent
        x_tensor = torch.tensor([x], dtype=torch.float32, device=(exp_val.device if isinstance(exp_val, torch.Tensor) else None))
        data['exponent_value_norm'] = x_tensor
        if self.replace_exponent:
            data['exponent_value'] = x_tensor

        # Overlap normalization (if target present)
        if 'target' in data and data['target'] is not None:
            target = data['target']
            # Determine the target row for current exponent if 2D
            if isinstance(target, torch.Tensor):
                tgt = target
                is_torch = True
            else:
                tgt = torch.tensor(target)
                is_torch = False

            if tgt.dim() == 2:
                exp_idx = int(data.get('exponent_index', 0))
                exp_idx = max(0, min(exp_idx, tgt.shape[0] - 1))
                vec = tgt[exp_idx].to(dtype=torch.float32)
            else:
                vec = tgt.to(dtype=torch.float32)

            # Determine n_atoms and per-atom basis length
            if 'atom_types' in data:
                n_atoms = int(len(data['atom_types']))
            else:
                n_atoms = -1
            if n_atoms <= 0:
                # Cannot safely normalize without atom count
                if self.replace_target:
                    data['target'] = vec
                else:
                    data['target_norm'] = vec
                return data

            M = int(vec.numel())
            if M % n_atoms != 0:
                # Unexpected shape; skip normalization
                if self.replace_target:
                    data['target'] = vec
                else:
                    data['target_norm'] = vec
                return data
            pab = M // n_atoms
            if pab < 25:
                # Not enough channels to split into L blocks; skip
                if self.replace_target:
                    data['target'] = vec
                else:
                    data['target_norm'] = vec
                return data

            # Compute per-L mu and std at exponent x
            mu_vals = [self._interp_1d(self.mean_grid[L], x) for L in range(5)]
            sd_vals = [max(self._interp_1d(self.std_grid[L], x), self.std_clip) for L in range(5)]

            # Reshape and apply normalization per L block
            arr = vec.view(n_atoms, pab).clone()
            for L, (s, t) in enumerate(L_SPANS):
                if t > pab:
                    break
                mu = float(mu_vals[L])
                sd = float(sd_vals[L])
                arr[:, s:t] = (arr[:, s:t] - mu) / sd
            vec_norm = arr.reshape(-1)

            if self.store_original:
                data['target_raw'] = vec
                data['overlap_norm_mu'] = torch.tensor(mu_vals, dtype=torch.float32)
                data['overlap_norm_std'] = torch.tensor(sd_vals, dtype=torch.float32)

            if self.replace_target:
                data['target'] = vec_norm
            else:
                data['target_norm'] = vec_norm

        return data


class RemoveOverlapOutliers:
    """
    Remove outlier values from the overlap target by zeroing any entries
    whose absolute value exceeds a provided cutoff.

    This transform is intended to be applied after normalization so that the
    cutoff is applied to normalized values. By default it will replace
    `data['target']` in-place. If `store_mask` is True the boolean mask of
    outliers will be stored in `data['target_outlier_mask']`.

    Assumptions made:
      - The overlap target lives in `data['target']` and is a torch.Tensor or
        numpy array. If it's numpy it will be converted to a tensor for
        processing.
      - Cutoff is a positive finite float; values with absolute value > cutoff
        are considered outliers and set to zero.
    """

    def __init__(self, cutoff: float = 1e6, replace_target: bool = True, store_mask: bool = True):
        self.cutoff = float(cutoff)
        self.replace_target = replace_target
        self.store_mask = store_mask

    def __call__(self, data: dict) -> dict:
        tgt = data.get('target')
        if tgt is None:
            return data

        # Convert to tensor if needed
        was_torch = True
        if not isinstance(tgt, torch.Tensor):
            was_torch = False
            tgt = torch.tensor(tgt, dtype=torch.float32)
        else:
            tgt = tgt.to(dtype=torch.float32)

        # Work with a copy to avoid unexpected in-place side-effects on original
        vec = tgt.clone()

        # Create mask of outliers (abs > cutoff)
        mask = torch.abs(vec) > float(self.cutoff)

        # Zero out outliers
        vec[mask] = 0.0

        # Store mask if requested
        if self.store_mask:
            data['target_outlier_mask'] = mask

        if self.replace_target:
            # Preserve device if original was tensor
            if was_torch and isinstance(data['target'], torch.Tensor):
                data['target'] = vec.to(device=data['target'].device)
            else:
                data['target'] = vec
        else:
            data['target_no_outliers'] = vec

        return data


class AddExponentEmbedding:
    """
    Prepare exponent information for the model.
    
    This ensures the exponent value is properly formatted for the neural network.
    """
    
    def __init__(self):
        pass

    def __call__(self, data):
        """
        Process exponent value.
        
        Expects data to have:
        - 'exponent_value': scalar or tensor
        
        Ensures:
        - 'exponent_value': properly shaped tensor
        """
        exp_val = data['exponent_value']
        
        # Ensure it's a tensor
        if not isinstance(exp_val, torch.Tensor):
            exp_val = torch.tensor(exp_val)
        
        # Ensure proper shape (scalar)
        if exp_val.dim() > 0:
            exp_val = exp_val.squeeze()
        if exp_val.dim() == 0:
            exp_val = exp_val.unsqueeze(0)
        
        data['exponent_value'] = exp_val
        return data


def create_overlap_transforms(
    radius: float = 6.0,
    max_neighbors: int = 50,
    device: Optional[torch.device] = None,
    include_self_edges: bool = False,
    norm_dir: Optional[Union[str, Path]] = None,
    replace_exponent: bool = True,
    replace_target: bool = True,
    outlier_cutoff: float = 1e6,
) -> MasterTransform:
    """
    Create a standard set of transforms for overlap prediction.
    
    This creates the graph structure expected by ESCN models.
    
    Args:
        radius: Cutoff distance for neighbor finding
        max_neighbors: Maximum number of neighbors per atom
        device: Target device for tensors
        include_self_edges: Whether to include self-connections
        
    Returns:
        MasterTransform with all necessary transforms
    """
    transforms: List = []
    # Step 1: Convert data to tensors
    transforms.append(ConvertToTensor(device=device))
    # Step 1.5: Apply normalization if configured
    if norm_dir is not None:
        transforms.append(
            OverlapNormalizationTransform(
                norm_dir=norm_dir,
                replace_exponent=replace_exponent,
                replace_target=replace_target,
            )
        )
        # Remove extreme outliers from normalized overlap targets
        transforms.append(RemoveOverlapOutliers(cutoff=outlier_cutoff, replace_target=replace_target))
    # Step 2: Add graph connectivity
    transforms.append(
        AddRadiusGraph(
            radius=radius,
            max_neighbors=max_neighbors,
            include_self_edges=include_self_edges,
        )
    )
    # Step 3: Compute edge vectors and distances
    transforms.append(AddEdgeVectorsAndDistances())
    # Step 4: Add batch indices
    transforms.append(AddBatchIndices(batch_idx=0))
    # Step 5: Process exponent information
    transforms.append(AddExponentEmbedding())
    
    return MasterTransform(transforms)


def create_overlap_data(mol, exp_idx: int, transforms):
    """
    Create overlap prediction data from a molecule and exponent index.
    
    Args:
        mol: CustomMolecule object
        exp_idx: Index of exponent to use
        transforms: Transform pipeline to apply
    
    Returns:
        Dictionary with processed data ready for model
    """
    # Create base data dictionary
    data = {
        'atom_types': mol.atom_types,
        'coords': mol.coords,
        'mol_id': mol.id,
        'n_atom': mol.n_atom,
        'n_vnode': mol.n_vnode,
        'exponent_index': exp_idx,
        'original_molecule': mol
    }
    
    # Add exponent value
    if mol.exponent_values is not None and exp_idx < len(mol.exponent_values):
        data['exponent_value'] = mol.exponent_values[exp_idx]
    else:
        data['exponent_value'] = 1.0  # Default
    
    # Add overlap target
    if hasattr(mol, 'overlap_int_2d') and mol.overlap_int_2d is not None:
        data['target'] = mol.overlap_int_2d
    
    # Apply transforms
    data = transforms(data)
    
    return data


def load_transform_config(config_path: Union[str, Path]) -> dict:
    """Load transform configuration from YAML file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_transforms_from_config(config_path: Union[str, Path]) -> MasterTransform:
    """Create transforms from configuration file."""
    config = load_transform_config(config_path)
    
    transforms = []
    for transform_config in config.get('transforms', []):
        target = transform_config.get('_target_')
        if not target:
            continue
            
        # Parse the target class
        module_path, class_name = target.rsplit('.', 1)
        
        # For our transforms, they should be in this module
        if module_path == 'scdp.model.overlap.transforms':
            transform_class = globals().get(class_name)
            if transform_class:
                # Get parameters (excluding _target_)
                params = {k: v for k, v in transform_config.items() if k != '_target_'}
                transforms.append(transform_class(**params))
    
    return MasterTransform(transforms)


def create_overlap_transforms_from_config(config_path: Optional[Union[str, Path]] = None, 
                                        radius: float = 6.0, 
                                        max_neighbors: int = 50) -> MasterTransform:
    """
    Create overlap transforms either from config file or default parameters.
    
    Args:
        config_path: Path to YAML configuration file. If None, use default parameters.
        radius: Cutoff radius for graph construction (used if config_path is None)
        max_neighbors: Maximum neighbors per atom (used if config_path is None)
    
    Returns:
        MasterTransform object with the sequence of transforms
    """
    if config_path is not None and Path(config_path).exists():
        return create_transforms_from_config(config_path)
    else:
        # Default transforms
        transforms = [
            ConvertToTensor(),
            AddRadiusGraph(radius=radius, max_neighbors=max_neighbors),
            AddEdgeVectorsAndDistances(),
            AddBatchIndices()
        ]
        return MasterTransform(transforms)

import torch
from typing import Dict, Optional, Tuple, List


def build_mapping_storage(mapping_info: Dict) -> Dict:
    """Create a serializable mapping_info_storage dict from mapping_info.

    Keeps keys used by the generator and converts exponent keys to strings.
    """
    return {
        'mapping': mapping_info.get('mapping'),
        'exponent_to_basis': {str(k): v for k, v in mapping_info.get('exponent_to_basis', {}).items()} if mapping_info.get('exponent_to_basis') else {},
        'basis_indices_mapping': mapping_info.get('basis_indices_mapping'),
        'basis_info': mapping_info.get('basis_info'),
        'exponent_values': mapping_info.get('exponent_values'),
        'atom_basis_structure': mapping_info.get('atom_basis_structure')
    }


def compute_nonpadded_stats(dense_overlap: Optional[torch.Tensor], basis_indices_mapping: Optional[List[List[int]]]) -> Dict[str, float]:
    """Compute sum/mean/std/min/max from the non-padded entries of dense_overlap.

    dense_overlap is expected to be a CPU/double tensor of shape (n_exponents, max_bfs_per_exp)
    and basis_indices_mapping lists the actual number of basis functions per exponent.
    If no values exist returns zeros.
    """
    stats = {'sum': 0.0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    if dense_overlap is None or not basis_indices_mapping:
        return stats

    all_vals = []
    for exp_idx, basis_list in enumerate(basis_indices_mapping):
        if basis_list:
            vals = dense_overlap[exp_idx, : len(basis_list)].flatten()
            all_vals.append(vals)

    if not all_vals:
        return stats

    concat = torch.cat(all_vals)
    stats['sum'] = float(concat.sum().item())
    stats['mean'] = float(concat.mean().item())
    stats['std'] = float(concat.std().item())
    stats['min'] = float(concat.min().item())
    stats['max'] = float(concat.max().item())
    return stats


def safe_shape_from_mapping(mapping_info_storage: Dict, dense_overlap_storage: Optional[torch.Tensor]) -> Optional[Tuple[int, int]]:
    """Return the (n_exponents, n_basis_functions) shape using basis_info or dense shape.
    Falls back to dense_overlap_storage shape when basis_info is not present.
    """
    if mapping_info_storage.get('basis_info'):
        bi = mapping_info_storage['basis_info']
        return (bi.get('n_exponents'), bi.get('total_basis_functions'))
    if dense_overlap_storage is not None:
        return tuple(dense_overlap_storage.shape)
    return None


def format_safe_molecule_id(molecule_id: str) -> str:
    """Sanitize molecule id for use in filenames while preserving underscores and zeros."""
    if molecule_id is None:
        return 'UNKNOWN'
    return str(molecule_id).replace('/', '_').replace('\\', '_').replace(':', '_')


def build_custom_molecule_from_compressed(
    atom_types,
    atom_coords,
    molecule_id: str,
    n_atom: int,
    basis_type: str,
    mapping_info_storage: Dict,
    mapping_info: Dict,
):
    """Construct a CompressedCustomMolecule and convert to the v2 CustomMolecule.

    This isolates the conversion logic so the generator stays compact.
    """
    from overlap_pred.compressed_custom_data import CompressedCustomMolecule
    import torch

    dense_overlap_matrix = mapping_info_storage.get('dense_overlap_matrix') if mapping_info_storage.get('dense_overlap_matrix') is not None else None
    # The generator stores dense_overlap separately; if not present the caller
    # should pass it via mapping_info_storage['dense_overlap_matrix'].
    basis_indices_mapping = mapping_info_storage.get('basis_indices_mapping')

    orig_shape = None
    if mapping_info_storage.get('basis_info'):
        bi = mapping_info_storage['basis_info']
        orig_shape = (bi.get('n_exponents'), bi.get('total_basis_functions'))

    exponent_values = mapping_info_storage.get('exponent_values')
    exponent_tensor = torch.as_tensor(exponent_values) if exponent_values is not None else None

    compressed = CompressedCustomMolecule(
        atom_types=atom_types.cpu(),
        coords=atom_coords.cpu(),
        id=molecule_id,
        metadata={},
        n_atom=int(n_atom),
        build_method=basis_type,
        dense_overlap_matrix=dense_overlap_matrix,
        basis_indices_mapping=basis_indices_mapping,
        exponent_values=exponent_tensor,
        original_sparse_shape=orig_shape,
        atom_basis_structure=mapping_info.get('atom_basis_structure'),
        basis_info=mapping_info.get('basis_info')
    )

    return compressed.to_custom_molecule_v2()

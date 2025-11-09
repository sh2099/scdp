import torch
from typing import Dict, List, Optional


def _build_header_lines(atom_basis_structure: Dict, atom_types: torch.Tensor, max_atoms_display: int, max_exponents_display: int, L_values_to_show: Optional[List[int]]):
    # Prepare header with three lines
    header_lines = ["", "", ""]
    header_lines[0] = f"{'Exp_Idx':<8}{'Exponent':<12}"
    for atom_idx in range(min(max_atoms_display, len(atom_basis_structure))):
        if atom_idx in atom_basis_structure:
            atom_type = atom_types[atom_idx].item()
            atom_struct = atom_basis_structure[atom_idx]
            total_cols = sum(len(range(-L, L+1)) for L in atom_struct.keys())
            atom_header = f"Atom_{atom_idx}(Z={atom_type})"
            header_lines[0] += f"{atom_header:^{total_cols*12}}"

    header_lines[1] = f"{'':^8}{'':^12}"
    for atom_idx in range(min(max_atoms_display, len(atom_basis_structure))):
        if atom_idx in atom_basis_structure:
            atom_struct = atom_basis_structure[atom_idx]
            for L in sorted(atom_struct.keys()):
                L_name = ['s', 'p', 'd', 'f', 'g', 'h'][L] if L < 6 else f'L{L}'
                n_m_values = 2*L + 1
                header_lines[1] += f"{L_name:^{n_m_values*12}}"

    header_lines[2] = f"{'':^8}{'':^12}"
    for atom_idx in range(min(max_atoms_display, len(atom_basis_structure))):
        if atom_idx in atom_basis_structure:
            atom_struct = atom_basis_structure[atom_idx]
            for L in sorted(atom_struct.keys()):
                for m in range(-L, L+1):
                    header_lines[2] += f"m={m:<9}"

    return header_lines


def _format_exp_row(exp_idx: int, exponent_values: List[float], atom_basis_structure: Dict, atom_types: torch.Tensor, overlap_2d: torch.Tensor, max_atoms_display: int, L_values_to_show: Optional[List[int]]):
    exp_val = exponent_values[exp_idx]
    row = f"{exp_idx:<8}{exp_val:<12.3e}"
    for atom_idx in range(min(max_atoms_display, len(atom_basis_structure))):
        if atom_idx in atom_basis_structure:
            atom_struct = atom_basis_structure[atom_idx]
            for L in sorted(atom_struct.keys()):
                for m in range(-L, L+1):
                    m_overlaps = []
                    if m in atom_struct[L]:
                        for basis_info_item in atom_struct[L][m]:
                            if basis_info_item['exp_idx'] == exp_idx:
                                basis_idx = basis_info_item['basis_idx']
                                overlap_val = overlap_2d[exp_idx, basis_idx].item()
                                m_overlaps.append(overlap_val)
                    if m_overlaps:
                        avg_overlap = sum(m_overlaps) / len(m_overlaps)
                        row += f"{avg_overlap:<12.3e}"
                    else:
                        row += f"{'---':<12}"
        else:
            row += f"{'---':<12}"
    return row


def _l_value_contribution_analysis(overlap_2d: torch.Tensor, original_atom_basis_structure: Dict, max_atoms_display: int):
    lines = []
    lines.append(f"{'Atom_Idx':<10}{'L':<5}{'L_Name':<8}{'N_Basis':<10}{'Total_Overlap':<15}{'Mean_Overlap':<15}")
    lines.append("-" * 78)
    for atom_idx in range(min(max_atoms_display, len(original_atom_basis_structure))):
        atom_struct = original_atom_basis_structure[atom_idx]
        for L in sorted(atom_struct.keys()):
            L_name = ['s', 'p', 'd', 'f', 'g', 'h'][L] if L < 6 else f'L{L}'
            L_basis_indices = []
            for m in atom_struct[L]:
                for basis_info_item in atom_struct[L][m]:
                    L_basis_indices.append(basis_info_item['basis_idx'])
            if L_basis_indices:
                L_total_overlap = overlap_2d[:, L_basis_indices].sum().item()
                L_mean_overlap = overlap_2d[:, L_basis_indices].mean().item()
                n_basis = len(L_basis_indices)
                lines.append(f"{atom_idx:<10}{L:<5}{L_name:<8}{n_basis:<10}{L_total_overlap:<15.3e}{L_mean_overlap:<15.3e}")
    return lines


def print_overlap_analysis_table(overlap_2d: torch.Tensor, mapping_info: Dict, atom_coords: torch.Tensor, atom_types: torch.Tensor, max_atoms_display: int = 5, max_exponents_display: int = 10, L_values_to_show: Optional[List[int]] = None):
    """Public function to print overlap table with L/m breakdown and L-value analysis.

    This function delegates to small helpers to build header and rows.
    """
    mapping = mapping_info['mapping']
    exponent_values = mapping_info['exponent_values']
    atom_basis_structure = mapping_info['atom_basis_structure']

    print(f"\n{'='*120}")
    print("OVERLAP INTEGRALS ANALYSIS WITH L,m BREAKDOWN")
    print(f"{'='*120}")
    print(f"Total exponents: {len(exponent_values)}")
    print(f"Total basis functions: {mapping_info['basis_info']['total_basis_functions']}")
    print(f"Total atoms: {mapping_info['basis_info']['n_atoms']}")

    if L_values_to_show is not None:
        print(f"Showing only L values: {L_values_to_show}")

    print(f"\nShowing first {max_atoms_display} atoms and {max_exponents_display} exponents:")
    header_lines = _build_header_lines(atom_basis_structure, atom_types, max_atoms_display, max_exponents_display, L_values_to_show)
    for line in header_lines:
        print(line)
    print("-" * len(header_lines[0]))

    for exp_idx in range(min(max_exponents_display, len(exponent_values))):
        row = _format_exp_row(exp_idx, exponent_values, atom_basis_structure, atom_types, overlap_2d, max_atoms_display, L_values_to_show)
        print(row)

    # L-value contribution analysis
    print(f"\n{'='*80}")
    print("L-VALUE CONTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    lines = _l_value_contribution_analysis(overlap_2d, mapping_info['atom_basis_structure'], max_atoms_display)
    for l in lines:
        print(l)

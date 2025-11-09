"""
Overlap Prediction ESCN Model

Based heavily on the ESCN (Equivariant Spherical Channel Network) architecture
for predicting molecular overlap integrals. This model:

1. Takes molecular structure (atom types + coordinates) + exponent value as i        # Initialize parameters
        self._initialize_parameters()2. Uses SO(3) equivariant layers similar to ESCN
3. Outputs overlap integrals in the overlap_int_2d format

Adaptations from ESCN:
- Added exponent embedding to include basis function exponent information
- Modified output head to predict overlap integrals instead of orbitals
- Simplified to focus on overlap prediction task
"""

import logging
import time
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn

from scdp.common.utils import get_edge_vectors_and_lengths, scatter
from scdp.model.scn.so3 import (
    CoefficientMapping,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
)
from scdp.model.scn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

# Import ESCN components we'll reuse
from scdp.model.scn.escn import LayerBlock, EdgeBlock


class OverlapESCN(nn.Module):
    """
    Overlap prediction model based on ESCN architecture.
    
    Key differences from ESCN:
    1. Includes exponent value as input
    2. Outputs exactly 25 overlap integrals per atom (following ESCN's per-atom approach)
    3. Uses spherical harmonic embeddings like ESCN for equivariance
    4. Fixed output dimension handles variable molecule sizes naturally
    
    Architecture:
    - Per-atom predictions: [N_atoms, 25] overlap integrals
    - Can be aggregated downstream as needed per molecule
    - Follows ESCN's pattern for handling variable-size molecules
    """
    
    def __init__(
        self,
        cutoff: float = 6.0,
        max_num_elements: int = 100,
        num_layers: int = 6,
        lmax_list: List[int] = [4],
        mmax_list: List[int] = [2],
        sphere_channels: int = 128,
        hidden_channels: int = 256,
        edge_channels: int = 128,
        num_sphere_samples: int = 128,
        distance_function: str = "gaussian",
        basis_width_scalar: float = 1.0,
        distance_resolution: float = 0.02,
        exponent_function: str = None,  # New: exponent embedding function (defaults to distance_function)
        exponent_min: float = 0.1,     # New: minimum exponent value for embedding range
        exponent_max: float = 10.0,    # New: maximum exponent value for embedding range
        show_timing_info: bool = False,
        *args, **kwargs
    ) -> None:
        super().__init__()

        self.cutoff = cutoff
        self.show_timing_info = show_timing_info
        self.max_num_elements = max_num_elements
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_sphere_samples = num_sphere_samples
        self.sphere_channels = sphere_channels
        self.edge_channels = edge_channels
        self.distance_resolution = distance_resolution
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions: int = len(self.lmax_list)
        self.basis_width_scalar = basis_width_scalar
        self.distance_function = distance_function
        self.exponent_function = exponent_function if exponent_function is not None else distance_function
        self.exponent_min = exponent_min
        self.exponent_max = exponent_max
        
        # Calculate sphere channels - no separate allocation needed now
        self.sphere_channels_all: int = (
            self.num_resolutions * self.sphere_channels
        )

        # Variables used for display purposes
        self.counter = 0

        # Non-linear activation function used throughout the network
        self.act = nn.SiLU()

        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels_all
        )

        # NEW: Exponent embedding using same approach as distance embedding
        # Maps exponent value to embedding space using distance embedding functions
        assert self.exponent_function in [
            "gaussian",
            "sigmoid", 
            "linearsigmoid",
            "silu",
        ]
        
        # Calculate number of exponent basis functions (similar to distance)
        self.num_exp_gaussians = int((self.exponent_max - self.exponent_min) / 0.1)  # 0.1 resolution
        if self.exponent_function == "gaussian":
            self.exponent_expansion = GaussianSmearing(
                self.exponent_min,
                self.exponent_max,
                self.num_exp_gaussians,
                self.basis_width_scalar,
            )
        elif self.exponent_function == "sigmoid":
            self.exponent_expansion = SigmoidSmearing(
                self.exponent_min,
                self.exponent_max,
                self.num_exp_gaussians,
                self.basis_width_scalar,
            )
        elif self.exponent_function == "linearsigmoid":
            self.exponent_expansion = LinearSigmoidSmearing(
                self.exponent_min,
                self.exponent_max,
                self.num_exp_gaussians,
                self.basis_width_scalar,
            )
        elif self.exponent_function == "silu":
            self.exponent_expansion = SiLUSmearing(
                self.exponent_min,
                self.exponent_max,
                self.num_exp_gaussians,
                self.basis_width_scalar,
            )
        
        # Linear projection from exponent embedding to sphere embedding space
        self.exponent_projection = nn.Linear(
            self.exponent_expansion.num_output,
            self.sphere_channels_all
        )

        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            "gaussian",
            "sigmoid",
            "linearsigmoid",
            "silu",
        ]
        self.num_gaussians = int(self.cutoff / self.distance_resolution)
        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                self.num_gaussians,
                self.basis_width_scalar,
            )
        elif self.distance_function == "sigmoid":
            self.distance_expansion = SigmoidSmearing(
                0.0,
                self.cutoff,
                self.num_gaussians,
                self.basis_width_scalar,
            )
        elif self.distance_function == "linearsigmoid":
            self.distance_expansion = LinearSigmoidSmearing(
                0.0,
                self.cutoff,
                self.num_gaussians,
                self.basis_width_scalar,
            )
        elif self.distance_function == "silu":
            self.distance_expansion = SiLUSmearing(
                0.0,
                self.cutoff,
                self.num_gaussians,
                self.basis_width_scalar,
            )

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = nn.ModuleList()
        for lval in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(SO3_Grid(lval, m))

            self.SO3_grid.append(SO3_m_grid)

        # Initialize the blocks for each layer of the GNN
        self.layer_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = LayerBlock(
                i,
                self.sphere_channels,
                self.hidden_channels,
                self.edge_channels,
                self.lmax_list,
                self.mmax_list,
                self.distance_expansion,
                self.max_num_elements,
                self.SO3_grid,
                self.act,
            )
            self.layer_blocks.append(block)

        # NEW: Overlap prediction head (following ESCN's approach)
        # Instead of the orbital readout, we predict 25 overlap integrals per atom
        # This ensures fixed output dimension regardless of molecule size
        self.overlap_per_atom = 25  # Fixed: 25 overlap integrals per atom
        
        # Calculate the input dimension for the readout layer
        # When we flatten x.embedding with shape [N_atoms, num_coeffs, sphere_channels]
        # we get [N_atoms, num_coeffs * sphere_channels]
        # where num_coeffs = sum((l+1)^2 for l in lmax_list) for each resolution
        total_coeffs = 0
        for lmax in self.lmax_list:
            total_coeffs += (lmax + 1) ** 2
        
        flattened_dim = total_coeffs * self.sphere_channels
        print(f"Calculated flattened dimension: {flattened_dim} (total_coeffs={total_coeffs}, sphere_channels={self.sphere_channels})")
        
        # Use a similar approach to ESCN's orbit_readout, but for overlap prediction
        # We'll use the full spherical harmonic embedding like ESCN does
        self.overlap_readout = nn.Sequential(
            nn.Linear(flattened_dim, self.hidden_channels),
            self.act,
            nn.Linear(self.hidden_channels, self.hidden_channels),
            self.act,
            nn.Linear(self.hidden_channels, self.overlap_per_atom),
        )

        # Initialize parameters
        self._initialize_parameters()
        
        # Convert model to double precision for better accuracy
        #self.double()

    def _initialize_parameters(self):
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def _init_edge_rot_mat(self, coords, edge_index, edge_distance_vec):
        """
        Initialize rotation matrices for edges.
        
        This method creates a local coordinate system for each edge by:
        1. Using the edge vector as the x-axis
        2. Creating orthogonal y and z axes using cross products
        3. Returning the rotation matrix that transforms global coords to edge-local coords
        
        This is critical for SO(3) equivariance - without proper edge rotation matrices,
        the spherical harmonic transformations will not be rotationally invariant.
        """
        edge_vec_0 = edge_distance_vec
        edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

        # Make sure the atoms are far enough apart
        if torch.min(edge_vec_0_distance) < 0.0001:
            import logging
            logging.error(
                "Error edge_vec_0_distance: {}".format(
                    torch.min(edge_vec_0_distance)
                )
            )
            (minval, minidx) = torch.min(edge_vec_0_distance, 0)
            logging.error(
                "Error edge_vec_0_distance: {} {} {} {} {}".format(
                    minidx,
                    edge_index[0, minidx],
                    edge_index[1, minidx],
                    coords[edge_index[0, minidx]],
                    coords[edge_index[1, minidx]],
                )
            )

        # Normalize edge vector to get x-axis
        norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

        # Create a random vector for constructing orthogonal axes
        edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
        edge_vec_2 = edge_vec_2 / (
            torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
        )
        
        # Create two rotated copies of the random vector in case it's aligned with norm_x
        # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
        edge_vec_2b = edge_vec_2.clone()
        edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
        edge_vec_2b[:, 1] = edge_vec_2[:, 0]
        edge_vec_2c = edge_vec_2.clone()
        edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
        edge_vec_2c[:, 2] = edge_vec_2[:, 1]
        
        # Choose the random vector that is least aligned with norm_x
        vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(-1, 1)
        vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(-1, 1)
        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
        
        edge_vec_2 = torch.where(
            torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2
        )
        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
        edge_vec_2 = torch.where(
            torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2
        )

        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
        # Check the vectors aren't aligned
        assert torch.max(vec_dot) < 0.99

        # Create orthogonal z-axis using cross product
        norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
        norm_z = norm_z / (
            torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True))
        )
        norm_z = norm_z / (
            torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1)
        )
        
        # Create y-axis using cross product
        norm_y = torch.cross(norm_x, norm_z, dim=1)
        norm_y = norm_y / (
            torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True))
        )

        # Construct the 3D rotation matrix
        norm_x = norm_x.view(-1, 3, 1)
        norm_y = -norm_y.view(-1, 3, 1)  # Note: negative y to match ESCN convention
        norm_z = norm_z.view(-1, 3, 1)

        edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
        edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

        return edge_rot_mat.detach()

    def forward(self, data):
        """
        Forward pass for overlap prediction.
        
        Expected data format (after transforms):
        - data['atom_types']: atomic numbers [N_atoms]
        - data['coords']: atomic coordinates [N_atoms, 3]
        - data['edge_index']: edge connectivity [2, N_edges]
        - data['edge_distance_vec']: edge vectors [N_edges, 3]
        - data['edge_distance']: edge distances [N_edges, 1]
        - data['shifts']: periodic boundary shifts [N_edges, 3]
        - data['exponent_value']: exponent value for this batch [scalar]
        - data['batch']: batch indices [N_atoms] indicating which molecule each atom belongs to
        
        Returns:
        - overlap_per_atom: [N_atoms, 25] tensor of overlap integrals per atom
          Following ESCN's pattern, downstream code can aggregate these per molecule
        """
        device = data['coords'].device
        dtype = data['coords'].dtype  # Use the dtype from input data
        atomic_numbers = data['atom_types'].long()
        num_atoms = len(atomic_numbers)

        if self.show_timing_info:
            start_time = time.time()

        # Get edge information (now provided by transforms)
        edge_index = data["edge_index"]
        edge_distance_vec = data["edge_distance_vec"] 
        edge_distance = data["edge_distance"].squeeze(-1)  # Remove last dimension if present

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data['coords'], edge_index, edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        self.SO3_edge_rot = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_edge_rot.append(
                SO3_Rotation(edge_rot_mat, self.lmax_list[i])
            )

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            device,
            dtype,
        )

        # Initialize the l=0,m=0 coefficients for each resolution
        offset_res = 0
        offset = 0
        for i in range(self.num_resolutions):
            # Base atomic embedding
            atomic_embedding = self.sphere_embedding(atomic_numbers)[
                :, offset : offset + self.sphere_channels
            ]
            
            # NEW: Add exponent embedding using distance-style expansion
            # Handle both single-molecule (scalar) and multi-molecule (vector) exponent values.
            exponent_value = data.get('exponent_value', torch.tensor(1.0, device=device, dtype=dtype))

            # Ensure exponent_value is 1D: [n_molecules] (or [1] for single-molecule)
            if isinstance(exponent_value, (float, int)):
                exponent_value = torch.tensor([exponent_value], device=device, dtype=dtype)
            if torch.is_tensor(exponent_value):
                exponent_value = exponent_value.to(device=device, dtype=dtype)
                if exponent_value.dim() == 0:
                    exponent_value = exponent_value.view(1)
                elif exponent_value.dim() > 1:
                    exponent_value = exponent_value.view(-1)

            # Expand / smear exponents -> projection
            exp_features = self.exponent_expansion(exponent_value)  # [n_molecules, num_exp_gaussians]
            exp_embedding = self.exponent_projection(exp_features)  # [n_molecules, sphere_channels_all]
            exp_embedding_slice = exp_embedding[:, offset : offset + self.sphere_channels]  # [n_molecules, sphere_channels]

            # Map per-molecule exponent embedding to atoms via batch indices
            if 'batch' in data and exponent_value.numel() > 1:
                batch_indices = data['batch']  # [num_atoms]
                # Safety: clamp indices if any mismatch (should not happen)
                if batch_indices.max().item() >= exp_embedding_slice.size(0):
                    raise ValueError(
                        f"Batch index {batch_indices.max().item()} exceeds exponent embedding count {exp_embedding_slice.size(0)}"
                    )
                per_atom_exp_embedding = exp_embedding_slice[batch_indices]  # [num_atoms, sphere_channels]
            else:
                # Single exponent value for whole structure
                per_atom_exp_embedding = exp_embedding_slice.expand(num_atoms, -1)

            # Combine atomic and exponent embeddings
            combined_embedding = atomic_embedding + per_atom_exp_embedding
            
            x.embedding[:, offset_res, :] = combined_embedding
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # This can be expensive to compute, so only do it once and pass it along to each layer
        mappingReduced = CoefficientMapping(
            self.lmax_list, self.mmax_list, device
        )

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        for i in range(self.num_layers):
            if i > 0:
                x_message = self.layer_blocks[i](
                    x,
                    atomic_numbers,
                    edge_distance,
                    edge_index,
                    self.SO3_edge_rot,
                    mappingReduced,
                )
                # Residual layer for all layers past the first
                x.embedding = x.embedding + x_message.embedding
            else:
                # No residual for the first layer
                x = self.layer_blocks[i](
                    x,
                    atomic_numbers,
                    edge_distance,
                    edge_index,
                    self.SO3_edge_rot,
                    mappingReduced,
                )

        ###############################################################
        # Predict overlap integrals (following ESCN's approach)
        ###############################################################
        
        # Flatten the spherical harmonic coefficients like ESCN does
        # Shape: [N_atoms, sphere_channels_all] -> flattens all l,m coefficients
        x_pt = x.embedding.flatten(1, 2)
        
        # Apply overlap readout to get per-atom overlap predictions
        # Shape: [N_atoms, 25] - exactly 25 overlap integrals per atom
        overlap_per_atom = self.overlap_readout(x_pt)
        
        # Return per-atom predictions - no aggregation here
        # The downstream code can aggregate these as needed per molecule
        # This follows ESCN's pattern of returning per-atom predictions

        if self.show_timing_info:
            end_time = time.time()
            print(f"OverlapESCN forward pass took {end_time - start_time:.4f} seconds")

        return overlap_per_atom


def aggregate_overlap_per_molecule(overlap_per_atom, batch_indices):
    """
    Utility function to aggregate per-atom overlap predictions into per-molecule predictions.
    
    Args:
        overlap_per_atom: [N_atoms, 25] tensor of overlap integrals per atom
        batch_indices: [N_atoms] tensor indicating which molecule each atom belongs to
        
    Returns:
        list of tensors: Each element is a flattened tensor [25 * n_atoms_in_mol] for each molecule
    """
    n_molecules = batch_indices.max().item() + 1
    overlap_per_molecule = []
    
    for mol_idx in range(n_molecules):
        # Get all atoms belonging to this molecule
        atom_mask = (batch_indices == mol_idx)
        mol_atom_predictions = overlap_per_atom[atom_mask]  # [n_atoms_in_mol, 25]
        # Flatten to get all overlap integrals for this molecule
        mol_overlap_flat = mol_atom_predictions.flatten()  # [25 * n_atoms_in_mol]
        overlap_per_molecule.append(mol_overlap_flat)
    
    return overlap_per_molecule

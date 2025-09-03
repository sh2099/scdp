# ESCN-Based Overlap Prediction Model

## Overview

This module implements an ESCN (Equivariant Spherical Channel Network) based model for predicting overlap integrals in quantum chemistry calculations. The model is specifically designed to predict molecular orbital overlap integrals as a function of basis function exponents, leveraging the rotational equivariance properties of ESCN for accurate molecular property prediction.

## Model Architecture

### Core Components

1. **OverlapESCN**: Main model class extending ESCN architecture
2. **ExponentEmbedding**: Custom embedding layer for basis function exponents
3. **OverlapReadout**: Specialized readout layer for overlap integral prediction
4. **Transform Pipeline**: Graph construction and data preprocessing

### Key Features

- **Equivariant Architecture**: Maintains rotational and translational invariance
- **Exponent Integration**: Incorporates basis function exponents as learnable features
- **Graph Neural Network**: Message passing over atomic neighborhoods
- **Spherical Harmonics**: E3-equivariant representations using SO3 operations

## Data Structure

### Input Data Format

The model works with `CustomMolecule` objects containing:

```python
class CustomMolecule:
    atom_types: torch.Tensor        # [n_atoms] - atomic numbers
    coords: torch.Tensor            # [n_atoms, 3] - atomic coordinates  
    exponent_values: torch.Tensor   # [n_exponents] - basis function exponents
    overlap_int_2d: torch.Tensor    # [n_exponents, overlap_dim] - target overlaps
    id: str                         # molecule identifier
    n_atom: int                     # number of atoms
    n_vnode: int                    # number of virtual nodes
```

### Data Dimensions

- **Molecules**: Typically 18 atoms (36 total including virtual nodes)
- **Exponents**: 18 different exponent values per molecule
- **Overlap Features**: 900-dimensional overlap integral vectors
- **Graph Edges**: ~1,200 edges generated with 6.0 Å cutoff

### Target Format

The overlap targets (`overlap_int_2d`) represent precomputed overlap integrals:
- Shape: `[n_exponents, 900]`
- Each row corresponds to one exponent value
- 900 features represent different orbital overlap combinations

## Data Loading Pipeline

### Transform System

The data loading uses a configurable transform pipeline:

```yaml
transforms:
  - ConvertToTensor: Convert numpy arrays to PyTorch tensors
  - AddRadiusGraph: Create graph connectivity (radius=6.0Å, max_neighbors=50)
  - AddEdgeVectorsAndDistances: Compute edge vectors and distances
  - AddBatchIndices: Add batch indexing for multi-molecule batches
```

### Key Transform Components

1. **ConvertToTensor**: Ensures all data is in tensor format
2. **AddRadiusGraph**: Creates edge connectivity based on atomic distances
3. **AddEdgeVectorsAndDistances**: Computes edge displacement vectors and distances
4. **AddBatchIndices**: Handles batching for training

### Dataset Structure

```python
OverlapDataset(
    data_path="/path/to/molecules",
    cutoff=6.0,                     # graph cutoff radius
    max_neighbors=50,               # max edges per atom
    max_samples_per_molecule=None,  # limit exponents per molecule
    transform=transform_pipeline    # preprocessing transforms
)
```

### Collate Function

The custom `overlap_collate_fn` handles:
- Batching multiple (molecule, exponent) pairs
- Adjusting edge indices for batched graphs
- Stacking exponent values and targets
- Creating proper batch indices for GNN message passing

## Model Training

### Training Configuration

```python
model = OverlapESCN(
    cutoff=6.0,                    # same as data cutoff
    max_num_elements=100,          # max atomic number
    num_layers=4,                  # ESCN layers
    lmax_list=[4, 4, 4, 4],       # spherical harmonic degrees
    mmax_list=[2, 2, 2, 2],       # spherical harmonic orders
    sphere_channels=128,           # spherical feature channels
    hidden_channels=256,           # hidden layer size
    edge_channels=128,             # edge feature channels
    num_sphere_samples=128,        # sphere sampling points
    exponent_channels=32,          # exponent embedding size
    overlap_output_dim=900,        # output dimension (matches targets)
    show_timing_info=False         # timing diagnostics
)
```

### Training Loop

```python
# Lightning module handles training
lightning_module = OverlapLightningModule(
    model=model,
    learning_rate=1e-3,
    weight_decay=1e-5
)

# DataLoader with custom collate function
dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=overlap_collate_fn,
    num_workers=4
)

# Training with PyTorch Lightning
trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=1
)
trainer.fit(lightning_module, dataloader)
```

### Loss Function

The model uses MSE loss between predicted and target overlap integrals:
```python
def compute_overlap_loss(outputs, targets):
    """Compute MSE loss for overlap prediction."""
    return F.mse_loss(outputs, targets)
```

## Model Forward Pass

### Input Processing

1. **Atom Embedding**: Embed atomic numbers into feature vectors
2. **Exponent Embedding**: Embed exponent values using learned embedding
3. **Graph Construction**: Use transforms to create edge connectivity
4. **Initial Features**: Combine atom and exponent embeddings

### ESCN Layers

Each ESCN layer performs:
1. **Message Passing**: Aggregate information from neighboring atoms
2. **Spherical Convolution**: Apply equivariant convolutions on sphere
3. **SO3 Nonlinearity**: Equivariant activation functions
4. **Layer Normalization**: Stabilize training

### Output Generation

1. **Global Pooling**: Aggregate atom-level features to molecule level
2. **Overlap Readout**: Project to 900-dimensional overlap space
3. **Final Output**: `[batch_size, 900]` overlap predictions

## Testing and Validation

### Test Suite

The model includes comprehensive tests:

```bash
python test_overlap_model.py
```

### Test Components

1. **Data Loading**: Verify dataset creation and transforms
2. **Model Creation**: Test model instantiation and parameter count
3. **Forward Pass**: Validate model execution with batched data
4. **Real Molecule**: Test with actual molecular data
5. **Single Molecule**: Debug individual molecule processing

### Performance Metrics

- **Forward Pass Time**: ~0.05 seconds per molecule
- **Model Parameters**: ~218k parameters (test configuration)
- **Memory Usage**: Efficient GPU memory utilization
- **Output Shapes**: Correct dimensional outputs `[batch_size, 900]`

### Expected Outputs

```
TESTING RESULTS:
- Dataset: 20 samples from 10 molecules
- Batch Processing: [2, 900] output for batch_size=2
- Single Molecule: [1, 900] output for individual prediction
- Edge Construction: ~1,200 edges for 36-atom molecules
- Execution Time: 0.05s per forward pass
```

## Configuration Files

### Transform Configuration

```yaml
# scdp/model/overlap/transform_config.yaml
_target_: scdp.model.overlap.transforms.create_overlap_transforms

transforms:
  - _target_: scdp.model.overlap.transforms.ConvertToTensor
  - _target_: scdp.model.overlap.transforms.AddRadiusGraph
    radius: 6.0
    max_num_neighbors: 50
  - _target_: scdp.model.overlap.transforms.AddEdgeVectorsAndDistances
  - _target_: scdp.model.overlap.transforms.AddBatchIndices

radius: 6.0
max_num_neighbors: 50
```

### Model Configuration

```yaml
# Example model config
model:
  _target_: scdp.model.overlap.ovlescn.OverlapESCN
  cutoff: 6.0
  num_layers: 4
  lmax_list: [4, 4, 4, 4]
  mmax_list: [2, 2, 2, 2]
  sphere_channels: 128
  hidden_channels: 256
  edge_channels: 128
  overlap_output_dim: 900
```

## Usage Examples

### Basic Usage

```python
from scdp.model.overlap import OverlapESCN, OverlapDataset, create_overlap_transforms_from_config

# Create dataset
dataset = OverlapDataset(
    data_path="/path/to/molecules",
    transform=create_overlap_transforms_from_config()
)

# Create model
model = OverlapESCN(overlap_output_dim=900)

# Forward pass
for batch in DataLoader(dataset, collate_fn=overlap_collate_fn):
    output = model(batch)  # [batch_size, 900]
```

### Training Script

```python
import pytorch_lightning as pl
from scdp.model.overlap import OverlapLightningModule

# Setup
model = OverlapESCN(overlap_output_dim=900)
lightning_module = OverlapLightningModule(model)
trainer = pl.Trainer(max_epochs=100)

# Train
trainer.fit(lightning_module, train_dataloader)
```

## File Structure

```
scdp/model/overlap/
├── __init__.py                 # Module exports
├── ovlescn.py                 # Main model implementation
├── module.py                  # Lightning training module
├── data_utils.py              # Dataset and data loading
├── transforms.py              # Data preprocessing transforms
├── transform_config.yaml     # Transform configuration
└── README.md                  # This documentation
```

## Dependencies

- PyTorch ≥ 1.12
- PyTorch Lightning ≥ 1.8
- e3nn ≥ 0.5.0 (for SO3 operations)
- torch-geometric ≥ 2.0 (for graph operations)
- numpy
- PyYAML (for configuration)

## Performance Considerations

### Memory Optimization
- Efficient batching with custom collate function
- Graph sparsity through radius cutoffs
- Gradient checkpointing for large models

### Computational Efficiency
- Fast sphere sampling in ESCN layers
- Optimized message passing
- GPU acceleration throughout pipeline

### Scalability
- Configurable model sizes
- Adjustable cutoff radii and neighbor limits
- Modular transform system for different data formats

## Future Extensions

### Potential Improvements
1. **Multi-scale graphs**: Different cutoffs for different interaction types
2. **Attention mechanisms**: Learnable attention weights for message passing  
3. **Transfer learning**: Pre-training on larger molecular datasets
4. **Uncertainty quantification**: Bayesian extensions for prediction uncertainty

### Research Directions
1. **Basis set optimization**: Joint optimization of exponents and overlap prediction
2. **Multi-property prediction**: Extend to other quantum chemical properties
3. **Active learning**: Intelligent selection of training molecules
4. **Interpretability**: Understanding what molecular features drive predictions

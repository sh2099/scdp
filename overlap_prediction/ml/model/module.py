"""
Lightning module for overlap integral prediction.

This module handles:
1. Loading CustomMolecule data (new format with overlap_int_2d)
2. Training the OverlapESCN model
3. Computing overlap-specific losses and metrics
4. Batching multiple exponents for the same molecule
"""

import math
import torch
import torch.nn.functional as F
from lightning import LightningModule
from hydra.utils import instantiate
from torch_ema import ExponentialMovingAverage
from typing import Dict, Any, Optional
import numpy as np

from scdp.common.utils import scatter


class OverlapLightningModule(LightningModule):
    """
    Overlap integral prediction with the OverlapESCN model.
    
    The data format expected:
    - Each batch contains multiple (molecule, exponent) pairs
    - For training efficiency, multiple exponents for the same molecule are batched together
    - Model predicts overlap integrals for the given exponent
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize the overlap prediction model
        self.model = instantiate(self.hparams.model)
        
        self.ema = ExponentialMovingAverage(
            self.parameters(), decay=self.hparams.train.ema.decay
        )        
        self.distributed = ((self.hparams.train.trainer.strategy == "ddp") and 
                            (self.hparams.train.trainer.devices > 1))
        
        # Register buffer for normalization if needed
        if hasattr(self.hparams, 'metadata') and 'target_var' in self.hparams.metadata:
            self.register_buffer("scale", torch.FloatTensor([self.hparams.metadata["target_var"]]).sqrt())
        else:
            self.register_buffer("scale", torch.FloatTensor([1.0]))

    def compute_overlap_loss(self, predictions, targets, exponent_indices):
        """
        Compute loss between predicted and target overlap integrals.
        
        Args:
            predictions: predicted overlap values (batch_size, n_atoms, total_basis_per_atom)
            targets: target overlap matrices (list of tensors, each with shape (n_exponents, n_atoms * total_basis_per_atom))
            exponent_indices: which exponent index to use for each prediction
            
        Returns:
            loss: computed loss value
        """
        total_loss = 0.0
        n_comparisons = 0
        
        batch_idx = 0
        for mol_idx, (target_matrix, exp_idx) in enumerate(zip(targets, exponent_indices)):
            if target_matrix is None:
                batch_idx += 1
                continue
                
            # Get the target row for this exponent
            if exp_idx < target_matrix.shape[0]:
                target_row = target_matrix[exp_idx, :]  # (n_atoms * total_basis_per_atom)
                
                # Get predictions for this molecule
                pred = predictions[batch_idx]  # (n_atoms, total_basis_per_atom)
                pred_flat = pred.flatten()  # (n_atoms * total_basis_per_atom)
                
                # Ensure they have the same length
                min_len = min(len(target_row), len(pred_flat))
                target_slice = target_row[:min_len]
                pred_slice = pred_flat[:min_len]
                
                # MSE loss
                loss = F.mse_loss(pred_slice, target_slice)
                total_loss += loss
                n_comparisons += 1
            
            batch_idx += 1
        
        return total_loss / max(n_comparisons, 1)

    def forward(self, batch):
        """Forward pass with overlap prediction."""
        # Batch should contain:
        # - molecule data (coords, atom_types, etc.)
        # - exponent_values (one per item in batch)
        # - target overlap matrices
        # - exponent indices (which exponent index each prediction corresponds to)
        
        overlap_predictions = self.model(batch)
        
        # Compute loss
        loss = self.compute_overlap_loss(
            overlap_predictions,
            batch.get('overlap_targets', []),
            batch.get('exponent_indices', [])
        )
        
        return {
            'predictions': overlap_predictions,
            'loss': loss
        }

    def training_step(self, batch, batch_idx):
        """Training step."""
        output = self.forward(batch)
        loss = output['loss']
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        output = self.forward(batch)
        loss = output['loss']
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {
            'val_loss': loss,
            'predictions': output['predictions']
        }

    def test_step(self, batch, batch_idx):
        """Test step."""
        output = self.forward(batch)
        loss = output['loss']
        
        return {
            'test_loss': loss,
            'predictions': output['predictions']
        }

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.train.optimizer.lr,
            weight_decay=self.hparams.train.optimizer.weight_decay,
            amsgrad=self.hparams.train.optimizer.amsgrad,
        )
        
        if hasattr(self.hparams.train, 'scheduler'):
            scheduler = instantiate(
                self.hparams.train.scheduler,
                optimizer=optimizer,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        
        return optimizer

    def on_before_backward(self, loss):
        """Update EMA before backward pass."""
        if self.ema is not None:
            self.ema.update(self.parameters())

    def predict_step(self, batch, batch_idx):
        """Prediction step for inference."""
        output = self.forward(batch)
        return {
            'predictions': output['predictions']
        }

import math
import torch
import torch.nn.functional as F
from lightning import LightningModule
from hydra.utils import instantiate
from torch_ema import ExponentialMovingAverage
from typing import Dict, Any, Optional
import numpy as np

from scdp.common.utils import scatter


class OverlapLightningModule(LightningModule):
    """
    Overlap integral prediction with the OverlapESCN model.
    
    Each input consists of:
    - A molecule (atom types + coordinates)
    - A single exponent value
    - Target overlap integrals for that exponent
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize the overlap prediction model
        self.model = instantiate(
            self.hparams.model,
            max_L=self.hparams.get('max_L', 4),
        )
        
        self.ema = ExponentialMovingAverage(
            self.parameters(), decay=self.hparams.train.ema.decay
        )        
        self.distributed = ((self.hparams.train.trainer.strategy == "ddp") and 
                            (self.hparams.train.trainer.devices > 1))
        
        # Register buffer for normalization if needed
        if hasattr(self.hparams, 'metadata') and 'target_var' in self.hparams.metadata:
            self.register_buffer("scale", torch.FloatTensor([self.hparams.metadata["target_var"]]).sqrt())
        else:
            self.register_buffer("scale", torch.FloatTensor([1.0]))

    def compute_overlap_loss(self, predictions, targets, target_indices=None):
        """
        Compute loss between predicted and target overlap integrals.
        
        Args:
            predictions: predicted overlap values (N_atoms, total_overlap_channels)
                        where total_overlap_channels = sum(2*L+1 for L in range(max_L+1))
            targets: target overlap values with same shape as predictions
            target_indices: optional indices to mask which predictions to compare
            
        Returns:
            loss: computed loss value
        """
        if targets is None:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Ensure targets have the same shape as predictions
        if targets.shape != predictions.shape:
            # Handle different target shapes - for now just use MSE on available data
            min_atoms = min(targets.shape[0], predictions.shape[0])
            min_channels = min(targets.shape[1], predictions.shape[1])
            targets_matched = targets[:min_atoms, :min_channels]
            predictions_matched = predictions[:min_atoms, :min_channels]
        else:
            targets_matched = targets
            predictions_matched = predictions
        
        # MSE loss
        loss = F.mse_loss(predictions_matched, targets_matched)
        return loss

    def forward(self, batch):
        """Forward pass with overlap prediction."""
        # Run model prediction
        overlap_predictions = self.model(batch)
        
        # Get targets if available
        targets = batch.get('overlap_targets', None)
        
        # Compute loss
        loss = self.compute_overlap_loss(overlap_predictions, targets)
        
        return {
            'predictions': overlap_predictions,
            'loss': loss,
            'batch': batch
        }

    def training_step(self, batch, batch_idx):
        """Training step."""
        output = self.forward(batch)
        loss = output['loss']
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        output = self.forward(batch)
        loss = output['loss']
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {
            'val_loss': loss,
            'predictions': output['predictions'],
            'batch': output['batch']
        }

    def test_step(self, batch, batch_idx):
        """Test step."""
        output = self.forward(batch)
        loss = output['loss']
        
        # Additional metrics could be computed here
        
        return {
            'test_loss': loss,
            'predictions': output['predictions'],
            'batch': output['batch']
        }

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.train.optimizer.lr,
            weight_decay=self.hparams.train.optimizer.weight_decay,
            amsgrad=self.hparams.train.optimizer.amsgrad,
        )
        
        if hasattr(self.hparams.train, 'scheduler'):
            scheduler = instantiate(
                self.hparams.train.scheduler,
                optimizer=optimizer,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        
        return optimizer

    def on_before_backward(self, loss):
        """Update EMA before backward pass."""
        if self.ema is not None:
            self.ema.update(self.parameters())

    def predict_step(self, batch, batch_idx):
        """Prediction step for inference."""
        output = self.forward(batch)
        return {
            'predictions': output['predictions'],
            'batch': output['batch']
        }

    def compute_overlap_metrics(self, predictions, batch):
        """
        Compute overlap-specific metrics.
        
        Args:
            predictions: model predictions
            batch: batch data
            
        Returns:
            metrics: dictionary of computed metrics
        """
        metrics = {}
        
        # Example metrics - can be extended
        if predictions is not None:
            # Compute average prediction magnitude
            avg_prediction = torch.mean(torch.abs(predictions))
            metrics['avg_prediction_magnitude'] = avg_prediction.item()
            
            # Compute prediction variance
            pred_var = torch.var(predictions)
            metrics['prediction_variance'] = pred_var.item()
            
            # Per-L analysis if we know the structure
            max_L = getattr(self.model, 'max_L', 4)
            start_idx = 0
            for L in range(max_L + 1):
                L_channels = 2 * L + 1
                end_idx = start_idx + L_channels
                L_preds = predictions[:, start_idx:end_idx]
                
                metrics[f'L{L}_mean'] = torch.mean(L_preds).item()
                metrics[f'L{L}_std'] = torch.std(L_preds).item()
                
                start_idx = end_idx
        
        return metrics

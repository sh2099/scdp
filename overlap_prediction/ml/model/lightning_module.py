"""
Lightning Module for Overlap Training

This module wraps the OverlapESCN model for Lightning training.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from ....scdp.model.overlap.ovlescn import OverlapESCN, aggregate_overlap_per_molecule

pylogger = logging.getLogger(__name__)


class OverlapLightningModule(pl.LightningModule):
    """
    Lightning module for overlap prediction training.
    
    Wraps the OverlapESCN model and handles:
    - Training, validation, and test steps
    - Loss computation and metrics
    - Optimizer and scheduler configuration
    - Logging and checkpointing
    """
    
    def __init__(
        self,
        # Model parameters
        cutoff: float = 6.0,
        # Optionally mask target overlap values above this cutoff so they do not
        # contribute to loss/metrics. If None, no masking is performed.
        mask_overlap_cutoff: Optional[float] = None,
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
        exponent_function: str = "gaussian",
        exponent_min: float = 0.0001,
        exponent_max: float = 100000.0,
        show_timing_info: bool = False,
        compile_model: bool = False,
        
        # Training parameters
        criterion: str = "mse",
        loss_aggregation: str = "mean",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        gradient_clip_val: float = 1.0,
        
        # Optimizer parameters
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        
        # Optional parameters for compatibility
        # Training configuration
        train: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        no_val_data: bool = False,  # Flag for when validation data is not available
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Store training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.criterion_name = criterion
        self.loss_aggregation = loss_aggregation
        
        # Store optimizer parameters
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        
        # Create model
        self.model = OverlapESCN(
            cutoff=cutoff,
            max_num_elements=max_num_elements,
            num_layers=num_layers,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            edge_channels=edge_channels,
            num_sphere_samples=num_sphere_samples,
            distance_function=distance_function,
            basis_width_scalar=basis_width_scalar,
            distance_resolution=distance_resolution,
            exponent_function=exponent_function,
            exponent_min=exponent_min,
            exponent_max=exponent_max,
            show_timing_info=show_timing_info,
        )
        
        # Ensure model is float32
        self.model = self.model.float()
        # Store masking cutoff for targets (None => no masking)
        self.mask_overlap_cutoff = mask_overlap_cutoff
        
        # Compile model if requested (PyTorch 2.0+)
        if compile_model and hasattr(torch, 'compile'):
            pylogger.info("Compiling model with torch.compile")
            self.model = torch.compile(self.model)
        
        # Loss function
        if criterion == "mse":
            self.criterion = nn.MSELoss()
        elif criterion == "mae":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        # Metrics
        self.train_mse = MeanSquaredError()
        self.train_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()
        self.test_mae = MeanAbsoluteError()
        
        # Store metadata
        self.metadata = metadata or {}
        
        # Store validation data availability flag
        self.no_val_data = no_val_data
        
        pylogger.info(f"Created OverlapLightningModule with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(batch)
    
    def compute_loss_and_predictions(
        self, 
        batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute loss and return predictions and targets for metrics.
        
        Returns:
            loss: Computed loss
            predictions: List of per-molecule predictions
            targets: List of per-molecule targets
        """
        # Forward pass - returns per-atom predictions [N_atoms, 25]
        overlap_per_atom = self.model(batch)
        
        # Aggregate per-atom predictions to per-molecule
        if 'batch' in batch:
            # Multi-molecule batch
            batch_indices = batch['batch']
            overlap_per_molecule = aggregate_overlap_per_molecule(overlap_per_atom, batch_indices)
        else:
            # Single molecule
            overlap_per_molecule = [overlap_per_atom.flatten()]
        
        # Extract targets
        if isinstance(batch['target'], list):
            # Variable-sized targets
            targets = batch['target']
            
            # Compute loss for each molecule
            losses = []
            valid_predictions = []
            valid_targets = []
            
            for i, target in enumerate(targets):
                if i < len(overlap_per_molecule):
                    pred = overlap_per_molecule[i]
                    # Apply masking of target values above cutoff if requested
                    if self.mask_overlap_cutoff is not None:
                        try:
                            mask = target <= float(self.mask_overlap_cutoff)
                        except Exception:
                            mask = (target <= self.mask_overlap_cutoff)

                        # If no valid entries after masking, skip this molecule
                        if mask.sum() == 0:
                            continue

                        masked_pred = pred[mask]
                        masked_target = target[mask]
                        mol_loss = self.criterion(masked_pred, masked_target)
                        valid_predictions.append(masked_pred)
                        valid_targets.append(masked_target)
                        losses.append(mol_loss)
                    else:
                        mol_loss = self.criterion(pred, target)
                        losses.append(mol_loss)
                        valid_predictions.append(pred)
                        valid_targets.append(target)
            
            if len(losses) > 0:
                if self.loss_aggregation == "mean":
                    loss = torch.stack(losses).mean()
                elif self.loss_aggregation == "sum":
                    loss = torch.stack(losses).sum()
                else:
                    raise ValueError(f"Unknown loss aggregation: {self.loss_aggregation}")
            else:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            return loss, valid_predictions, valid_targets
        else:
            # Fixed-size targets
            # When a mask cutoff is provided, compute per-sample masked losses
            predictions = [pred for pred in overlap_per_molecule]
            targets_tensor = batch['target']

            if self.mask_overlap_cutoff is None:
                predictions_tensor = torch.stack(overlap_per_molecule)
                loss = self.criterion(predictions_tensor, targets_tensor)
                # Convert targets to list format
                targets_list = [targets_tensor[i] for i in range(len(predictions))]
                return loss, predictions, targets_list
            else:
                # Compute per-sample masked losses similar to variable-sized case
                losses = []
                valid_predictions = []
                valid_targets = []

                for i, pred in enumerate(overlap_per_molecule):
                    target = targets_tensor[i]
                    try:
                        mask = target <= float(self.mask_overlap_cutoff)
                    except Exception:
                        mask = (target <= self.mask_overlap_cutoff)

                    if mask.sum() == 0:
                        continue

                    masked_pred = pred[mask]
                    masked_target = target[mask]
                    losses.append(self.criterion(masked_pred, masked_target))
                    valid_predictions.append(masked_pred)
                    valid_targets.append(masked_target)

                if len(losses) > 0:
                    if self.loss_aggregation == "mean":
                        loss = torch.stack(losses).mean()
                    elif self.loss_aggregation == "sum":
                        loss = torch.stack(losses).sum()
                    else:
                        raise ValueError(f"Unknown loss aggregation: {self.loss_aggregation}")
                else:
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                return loss, valid_predictions, valid_targets
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss, predictions, targets = self.compute_loss_and_predictions(batch)
        
        # Calculate effective batch size (number of molecules)
        batch_size = len(predictions)
        
        # Update metrics
        for pred, target in zip(predictions, targets):
            self.train_mse(pred, target)
            self.train_mae(pred, target)
        
        # Log metrics with explicit batch size
        # Use sync_dist only if we're actually in distributed mode and it's safe
        sync_dist = self.trainer.world_size > 1 if self.trainer else False
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=sync_dist, batch_size=batch_size)
        self.log("train_mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=sync_dist, batch_size=batch_size)
        self.log("train_mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=sync_dist, batch_size=batch_size)
        
        # Log learning rate properly for distributed training
        if sync_dist:
            self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], 
                    on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        loss, predictions, targets = self.compute_loss_and_predictions(batch)
        
        # Calculate effective batch size (number of molecules)
        batch_size = len(predictions)
        
        # Update metrics
        for pred, target in zip(predictions, targets):
            self.val_mse(pred, target)
            self.val_mae(pred, target)
        
        # Log metrics with explicit batch size
        # Use sync_dist only if we're actually in distributed mode and it's safe
        sync_dist = self.trainer.world_size > 1 if self.trainer else False
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=sync_dist, batch_size=batch_size)
        self.log("val_mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=sync_dist, batch_size=batch_size)
        self.log("val_mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=sync_dist, batch_size=batch_size)
        
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Test step."""
        loss, predictions, targets = self.compute_loss_and_predictions(batch)
        
        # Calculate effective batch size (number of molecules)
        batch_size = len(predictions)
        
        # Update metrics
        for pred, target in zip(predictions, targets):
            self.test_mse(pred, target)
            self.test_mae(pred, target)
        
        # Log metrics with explicit batch size
        # Use sync_dist only if we're actually in distributed mode and it's safe
        sync_dist = self.trainer.world_size > 1 if self.trainer else False
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=sync_dist, batch_size=batch_size)
        self.log("test_mse", self.test_mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=sync_dist, batch_size=batch_size)
        self.log("test_mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=sync_dist, batch_size=batch_size)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(self.adam_beta1, self.adam_beta2),
            amsgrad=True,
            eps=1e-8
        )
        
        # Determine scheduler monitor based on data availability
        monitor_metric = "val_loss"
        if hasattr(self, 'no_val_data') and self.no_val_data:
            monitor_metric = "train_loss"
            pylogger.info(f"Using {monitor_metric} for scheduler monitoring (no validation data)")
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.8,
            patience=10,
            threshold=1e-6,
            min_lr=1e-7
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": monitor_metric,
                "frequency": 1,
                "strict": False,  # Don't fail if metric is missing
            }
        }
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr, on_epoch=True, prog_bar=False, sync_dist=False)
    
    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> List[torch.Tensor]:
        """Prediction step for inference."""
        with torch.no_grad():
            overlap_per_atom = self.model(batch)
            
            # Aggregate per-atom predictions to per-molecule
            if 'batch' in batch:
                batch_indices = batch['batch']
                overlap_per_molecule = aggregate_overlap_per_molecule(overlap_per_atom, batch_indices)
            else:
                overlap_per_molecule = [overlap_per_atom.flatten()]
            
            return overlap_per_molecule

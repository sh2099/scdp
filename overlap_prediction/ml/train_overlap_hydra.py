#!/usr/bin/env python3
"""
Hydra-based Full Scale Overlap Training Script

This script uses Hydra configuration management for training the OverlapESCN model
on the complete molecular dataset with proper configuration management.

Usage:
    # Single GPU with default config
    python train_overlap_hydra.py

    # Override configuration options
    python train_overlap_hydra.py trainer.devices=2 data.batch_size=32

    # Use custom config
    python train_overlap_hydra.py --config-name overlap_full

    # Multi-GPU with specific GPUs
    python train_overlap_hydra.py trainer.devices=[0,1,2,3] trainer.strategy=ddp

    # Quick test with fewer epochs
    python train_overlap_hydra.py trainer.max_epochs=5 data.batch_size=8
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import hydra
import lightning.pytorch as pl
import omegaconf
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig

# Add the project root to Python path
sys.path.append('/export/home/hmichael/scdp')

from scdp.common.system import log_hyperparameters, PROJECT_ROOT
from scdp.model.overlap import OverlapDataModule, OverlapLightningModule

# Disable SLURM detection for Lightning
from lightning.pytorch.plugins.environments import SLURMEnvironment
SLURMEnvironment.detect = lambda: False

pylogger = logging.getLogger(__name__)


class CheckpointTimingCallback(Callback):
    """Callback to track checkpoint timing."""
    
    def __init__(self):
        super().__init__()
        self.checkpoint_start = None
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self.checkpoint_start = time.time()
        pylogger.info(f"💾 Starting checkpoint save...")
    
    def on_train_epoch_end(self, trainer, pl_module):
        # Monitor for checkpoint completion after checkpointing
        if self.checkpoint_start:
            duration = time.time() - self.checkpoint_start
            pylogger.info(f"💾 Checkpoint processing took {duration:.1f}s")
            self.checkpoint_start = None


class EpochTimingCallback(Callback):
    """Callback to track and log epoch timing information."""
    
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
        self.train_start_time = None
        self.val_start_time = None
        self.last_timestamp = None
    
    def on_train_epoch_start(self, trainer, pl_module):
        current_time = time.time()
        if self.last_timestamp:
            gap = current_time - self.last_timestamp
            pylogger.info(f"⏰ Gap since last activity: {gap:.1f}s")
        
        self.epoch_start_time = current_time
        self.train_start_time = current_time
        pylogger.info(f"🏃 Epoch {trainer.current_epoch} - Training phase started")
    
    def on_validation_epoch_start(self, trainer, pl_module):
        current_time = time.time()
        if self.train_start_time:
            train_duration = current_time - self.train_start_time
            pylogger.info(f"✅ Epoch {trainer.current_epoch} - Training completed in {train_duration:.1f}s")
        
        self.val_start_time = current_time
        pylogger.info(f"🔍 Epoch {trainer.current_epoch} - Validation phase started")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        current_time = time.time()
        if self.val_start_time:
            val_duration = current_time - self.val_start_time
            pylogger.info(f"✅ Epoch {trainer.current_epoch} - Validation completed in {val_duration:.1f}s")
    
    def on_train_epoch_end(self, trainer, pl_module):
        current_time = time.time()
        if self.epoch_start_time:
            epoch_duration = current_time - self.epoch_start_time
            pylogger.info(f"🎯 Epoch {trainer.current_epoch} - Total epoch time: {epoch_duration:.1f}s")
        
        pylogger.info(f"🔧 Starting epoch-end callbacks and checkpointing...")
        self.last_timestamp = current_time
    
    def on_train_end(self, trainer, pl_module):
        current_time = time.time()
        if self.last_timestamp:
            gap = current_time - self.last_timestamp
            pylogger.info(f"⏰ Time spent in post-epoch processing: {gap:.1f}s")
        pylogger.info(f"🏁 Training completely finished")


# Set PyTorch matmul precision for better performance
torch.set_float32_matmul_precision("high")


def build_callbacks(cfg: omegaconf.ListConfig) -> list:
    """Instantiate callbacks from configuration."""
    callbacks = []
    
    for callback_cfg in cfg:
        pylogger.info(f"Adding callback <{callback_cfg['_target_'].split('.')[-1]}>")
        callback = hydra.utils.instantiate(callback_cfg, _recursive_=False)
        callbacks.append(callback)
    
    return callbacks


def setup_logger(cfg: DictConfig) -> Optional[pl.loggers.Logger]:
    """Setup experiment logger."""
    
    # Use TensorBoard logger
    if "tensorboard" in cfg.train.logging:
        pylogger.info("Setting up TensorBoard logger")
        tb_config = cfg.train.logging.tensorboard
        logger = TensorBoardLogger(
            save_dir=tb_config.save_dir,
            name=tb_config.name
        )
        return logger
    else:
        pylogger.warning("No TensorBoard logger specified in configuration")
        return None


def run_training(cfg: DictConfig) -> str:
    """
    Main training function.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Path to the output directory
    """
    
    # Set random seed for reproducibility
    if cfg.train.get("seed"):
        seed_everything(cfg.train.seed, workers=True)
    
    # Setup storage directory
    storage_dir = cfg.core.storage_dir
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    pylogger.info(f"Output directory: {storage_dir}")
    
    # Save configuration
    yaml_conf = omegaconf.OmegaConf.to_yaml(cfg)
    (Path(storage_dir) / "config.yaml").write_text(yaml_conf)
    
    # Instantiate data module
    pylogger.info(f"Instantiating data module <{cfg.data['_target_']}>")
    datamodule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    
    # Setup data module to get metadata
    datamodule.setup(stage="fit")
    metadata = getattr(datamodule, "metadata", None)
    
    if metadata is None:
        pylogger.warning(f"No metadata found in data module")
        metadata = {}
    
    # Save metadata
    import json
    with open(Path(storage_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Instantiate model
    pylogger.info(f"Instantiating model <{cfg.model['_target_']}>")
    
    # Check for validation data availability
    has_val_data = hasattr(datamodule, 'val_dataset') and datamodule.val_dataset is not None
    if has_val_data:
        val_size = len(datamodule.val_dataset) if datamodule.val_dataset else 0
        has_val_data = val_size > 0
    
    if not has_val_data:
        pylogger.warning("No validation data - scheduler will monitor train_loss")
    
    model = hydra.utils.instantiate(
        cfg.model, 
        train=cfg.train,
        metadata=metadata,
        no_val_data=not has_val_data,  # Pass as constructor argument
        _recursive_=False
    )
    
    # Build callbacks
    callbacks = build_callbacks(cfg.train.callbacks)
    
    # Add timing callbacks for performance monitoring
    timing_callback = EpochTimingCallback()
    checkpoint_timing_callback = CheckpointTimingCallback()
    callbacks.extend([timing_callback, checkpoint_timing_callback])
    
    # Check if we have validation data, and adjust callbacks if needed
    if hasattr(datamodule, 'val_dataset') and datamodule.val_dataset is not None:
        val_size = len(datamodule.val_dataset) if datamodule.val_dataset else 0
        if val_size == 0:
            pylogger.warning("No validation data available - adjusting callbacks for training-only mode")
            
            # Remove or modify callbacks that depend on validation metrics
            adjusted_callbacks = []
            for callback in callbacks:
                if hasattr(callback, 'monitor') and 'val_' in str(callback.monitor):
                    if 'EarlyStopping' in str(type(callback)):
                        pylogger.info("Replacing EarlyStopping val_loss monitor with train_loss")
                        callback.monitor = 'train_loss'
                    elif 'ModelCheckpoint' in str(type(callback)):
                        pylogger.info("Replacing ModelCheckpoint val_loss monitor with train_loss")
                        callback.monitor = 'train_loss'
                adjusted_callbacks.append(callback)
            callbacks = adjusted_callbacks
    
    # Setup logger
    logger = setup_logger(cfg)
    
    # Check for existing checkpoint
    ckpt_path = None
    last_ckpt = Path(storage_dir) / "last.ckpt"
    if last_ckpt.exists():
        ckpt_path = str(last_ckpt)
        pylogger.info(f"Found existing checkpoint: {ckpt_path}")
    else:
        # Look for epoch checkpoints
        ckpt_files = list(Path(storage_dir).glob("*epoch*.ckpt"))
        if ckpt_files:
            # Get most recent checkpoint
            ckpt_files.sort(key=lambda x: x.stat().st_mtime)
            ckpt_path = str(ckpt_files[-1])
            pylogger.info(f"Found epoch checkpoint: {ckpt_path}")
    
    # Create trainer
    trainer_cfg = cfg.train.trainer.copy()
    
    # Add distributed training specific configurations
    if trainer_cfg.get('strategy') == 'ddp':
        pylogger.info("Configuring DDP strategy for distributed training")
        
        # Keep strategy as string but set up DDP-specific options
        # Lightning will handle the DDPStrategy creation internally
        trainer_cfg['strategy'] = 'ddp'
        
        # Set process group timeout via environment variable
        import os
        if 'LOCAL_RANK' in os.environ:
            pylogger.info("Distributed environment detected")
            # Set a reasonable timeout for distributed operations
            os.environ['TORCH_DISTRIBUTED_TIMEOUT'] = '1800'  # 30 minutes
            
            # Suppress NCCL device warnings by setting proper CUDA device
            import torch
            if torch.cuda.is_available():
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                torch.cuda.set_device(local_rank)
    
    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        logger=logger,
        callbacks=callbacks,
        **trainer_cfg
    )
    
    # Log hyperparameters safely
    try:
        # Create a safe version of the config for logging
        safe_config = {}
        
        # Add basic configuration info
        safe_config['model_name'] = cfg.model._target_.split('.')[-1]
        safe_config['data_module'] = cfg.data._target_.split('.')[-1]
        safe_config['batch_size'] = cfg.data.batch_size
        safe_config['learning_rate'] = cfg.model.learning_rate
        safe_config['max_epochs'] = cfg.train.trainer.max_epochs
        safe_config['num_layers'] = cfg.model.num_layers
        safe_config['hidden_channels'] = cfg.model.hidden_channels
        safe_config['sphere_channels'] = cfg.model.sphere_channels
        safe_config['cutoff'] = cfg.model.cutoff
        
        if trainer.logger:
            trainer.logger.log_hyperparams(safe_config)
            pylogger.info("Hyperparameters logged successfully")
    except Exception as e:
        pylogger.warning(f"Could not log hyperparameters: {e}")
    
    # Save full config to file
    yaml_conf = omegaconf.OmegaConf.to_yaml(cfg)
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    (Path(storage_dir) / "config.yaml").write_text(yaml_conf)
    
    # Start training
    pylogger.info("Starting training...")
    pylogger.info("🔄 Phase: trainer.fit() - beginning")
    
    import time
    start_time = time.time()
    
    try:
        trainer.fit(
            model, 
            datamodule=datamodule,
            ckpt_path=ckpt_path
        )
        
        end_time = time.time()
        duration = end_time - start_time
        pylogger.info(f"✅ Phase: trainer.fit() - completed successfully in {duration:.1f}s")
        
    except Exception as e:
        pylogger.error(f"❌ Phase: trainer.fit() - failed with error: {e}")
        raise
    
    # Test on best model if test data available
    if datamodule.test_dataset is not None:
        pylogger.info("Starting testing...")
        trainer.test(datamodule=datamodule, ckpt_path="best")
    
    # Close logger
    if logger:
        pylogger.info("Closing logger")
    
    pylogger.info(f"Training completed. Results saved to: {storage_dir}")
    return storage_dir


@hydra.main(
    config_path=str(PROJECT_ROOT / "scdp" / "config"), 
    config_name="overlap_full", 
    version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    
    # Configure Hydra to not change working directory 
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    
    # Set up logging
    pylogger.info("🚀 Starting Overlap Training with Hydra")
    pylogger.info(f"Configuration:\n{omegaconf.OmegaConf.to_yaml(cfg)}")
    
    # Check for GPU availability
    if not torch.cuda.is_available():
        pylogger.warning("CUDA not available. Training will use CPU.")
    else:
        pylogger.info(f"CUDA available. Found {torch.cuda.device_count()} GPUs.")
    
    try:
        # Run training
        output_dir = run_training(cfg)
        pylogger.info(f"✅ Training completed successfully!")
        pylogger.info(f"📁 Results saved to: {output_dir}")
        
    except Exception as e:
        pylogger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

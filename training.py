"""
Phase 4: Training Implementation for UV-Vis Spectra PINN
=======================================================

This module implements the multi-stage training protocol for Physics-Informed Neural Networks
applied to UV-Vis spectroscopy data, including:
- Multi-stage training with adaptive optimizer switching
- Leave-one-scan-out cross-validation
- Comprehensive monitoring and callbacks
- Model checkpointing and recovery

Multi-Stage Training Protocol:
1. Small weight initialization using DeepXDE patterns
2. Adam optimizer (lr=1e-3) for initial training
3. L-BFGS optimizer for fine-tuning until convergence
4. Cross-validation: Leave-one-scan-out validation across concentration levels

Author: Claude Code Assistant
"""

import numpy as np
import deepxde as dde
import tensorflow as tf
from typing import Tuple, Optional, Callable, Dict, Any, List, Union
import logging
import os
import json
import time
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

from .data_preprocessing import UVVisDataProcessor
from .model_definition import SpectroscopyPINN
from .loss_functions import UVVisLossFunction, create_uvvis_loss_function

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UVVisTrainingStrategy:
    """
    Multi-stage training strategy for UV-Vis spectroscopy PINN.
    
    This class implements adaptive training with automatic optimizer switching,
    comprehensive monitoring, and robust convergence detection.
    """
    
    def __init__(self, 
                 model: dde.Model,
                 loss_function: UVVisLossFunction,
                 checkpoint_dir: str = "checkpoints",
                 enable_adaptive_switching: bool = True,
                 convergence_patience: int = 1000,
                 min_improvement_delta: float = 1e-6):
        """
        Initialize the training strategy.
        
        Args:
            model: DeepXDE model to train
            loss_function: UV-Vis loss function instance
            checkpoint_dir: Directory for saving checkpoints
            enable_adaptive_switching: Whether to use adaptive optimizer switching
            convergence_patience: Patience for convergence detection
            min_improvement_delta: Minimum improvement for convergence
        """
        self.model = model
        self.loss_function = loss_function
        self.checkpoint_dir = Path(checkpoint_dir)
        self.enable_adaptive_switching = enable_adaptive_switching
        self.convergence_patience = convergence_patience
        self.min_improvement_delta = min_improvement_delta
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state tracking
        self.current_stage = 'initialization'
        self.training_history = {}
        self.metrics_tracker = TrainingMetricsTracker()
        self.best_loss = float('inf')
        self.best_model_path = None
        
        # Stage configurations
        self.stage_configs = self._create_stage_configurations()
        
        logger.info(f"Initialized training strategy with adaptive switching: {enable_adaptive_switching}")
    
    def _create_stage_configurations(self) -> Dict[str, Dict]:
        """Create configuration for each training stage."""
        return {
            'initialization': {
                'optimizer': 'adam',
                'learning_rate': 1e-4,
                'max_iterations': 1000,
                'description': 'Small weight initialization and warm-up',
                'early_stopping_patience': 500,
                'weight_scale': 0.1
            },
            'adam_training': {
                'optimizer': 'adam',
                'learning_rate': 1e-3,
                'max_iterations': 15000,
                'description': 'Adam optimization phase',
                'early_stopping_patience': 2000,
                'lr_decay_factor': 0.9,
                'lr_decay_steps': 1000
            },
            'lbfgs_refinement': {
                'optimizer': 'L-BFGS',
                'learning_rate': None,  # L-BFGS auto-adjusts
                'max_iterations': 10000,
                'description': 'L-BFGS fine-tuning phase',
                'early_stopping_patience': 500,
                'convergence_tolerance': 1e-8
            }
        }
    
    def execute_full_training(self) -> Dict[str, Any]:
        """
        Execute the complete multi-stage training protocol.
        
        Returns:
            Training results and statistics
        """
        logger.info("Starting multi-stage UV-Vis PINN training...")
        start_time = time.time()
        
        try:
            # Stage 1: Initialization
            self._execute_initialization_stage()
            
            # Stage 2: Adam Training
            self._execute_adam_stage()
            
            # Stage 3: L-BFGS Refinement
            self._execute_lbfgs_stage()
            
            # Final evaluation and cleanup
            final_results = self._finalize_training()
            
            total_time = time.time() - start_time
            logger.info(f"Multi-stage training completed in {total_time:.2f} seconds")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return self._create_failure_result(str(e))
    
    def _execute_initialization_stage(self) -> None:
        """Execute initialization stage with small weights."""
        logger.info("Stage 1: Initialization with small weights...")
        self.current_stage = 'initialization'
        config = self.stage_configs['initialization']
        
        # Apply small weight initialization
        self._initialize_small_weights(config['weight_scale'])
        
        # Compile model for initialization stage
        self.model.compile(
            optimizer='adam',
            lr=config['learning_rate']
        )
        
        # Create callbacks for initialization
        callbacks = self._create_stage_callbacks('initialization')
        
        # Train initialization stage
        losshistory = self.model.train(
            iterations=config['max_iterations'],
            callbacks=callbacks,
            display_every=200
        )
        
        # Store results
        self.training_history['initialization'] = {
            'losshistory': losshistory,
            'config': config,
            'final_loss': losshistory.loss_train[-1] if losshistory.loss_train else float('inf')
        }
        
        logger.info(f"Initialization stage completed. Final loss: {self.training_history['initialization']['final_loss']:.6f}")
    
    def _execute_adam_stage(self) -> None:
        """Execute Adam optimization stage."""
        logger.info("Stage 2: Adam optimization...")
        self.current_stage = 'adam_training'
        config = self.stage_configs['adam_training']
        
        # Compile model for Adam stage
        self.model.compile(
            optimizer='adam',
            lr=config['learning_rate']
        )
        
        # Create callbacks with learning rate scheduling
        callbacks = self._create_stage_callbacks('adam_training')
        
        # Add learning rate scheduler if specified
        if 'lr_decay_factor' in config:
            lr_scheduler = LearningRateScheduler(
                initial_lr=config['learning_rate'],
                decay_factor=config['lr_decay_factor'],
                decay_steps=config['lr_decay_steps']
            )
            callbacks.append(lr_scheduler)
        
        # Train Adam stage with potential early switching
        if self.enable_adaptive_switching:
            losshistory = self._train_with_adaptive_switching(config, callbacks)
        else:
            losshistory = self.model.train(
                iterations=config['max_iterations'],
                callbacks=callbacks,
                display_every=500
            )
        
        # Store results
        self.training_history['adam_training'] = {
            'losshistory': losshistory,
            'config': config,
            'final_loss': losshistory.loss_train[-1] if losshistory.loss_train else float('inf')
        }
        
        logger.info(f"Adam stage completed. Final loss: {self.training_history['adam_training']['final_loss']:.6f}")
    
    def _execute_lbfgs_stage(self) -> None:
        """Execute L-BFGS refinement stage."""
        logger.info("Stage 3: L-BFGS refinement...")
        self.current_stage = 'lbfgs_refinement'
        config = self.stage_configs['lbfgs_refinement']
        
        # Compile model for L-BFGS stage
        self.model.compile(optimizer='L-BFGS')
        
        # Create callbacks for L-BFGS
        callbacks = self._create_stage_callbacks('lbfgs_refinement')
        
        # Train L-BFGS stage
        losshistory = self.model.train(
            iterations=config['max_iterations'],
            callbacks=callbacks,
            display_every=100
        )
        
        # Store results
        self.training_history['lbfgs_refinement'] = {
            'losshistory': losshistory,
            'config': config,
            'final_loss': losshistory.loss_train[-1] if losshistory.loss_train else float('inf')
        }
        
        logger.info(f"L-BFGS stage completed. Final loss: {self.training_history['lbfgs_refinement']['final_loss']:.6f}")
    
    def _initialize_small_weights(self, weight_scale: float) -> None:
        """Initialize network with small weights."""
        try:
            # Get the neural network from the model
            net = self.model.net
            
            # Custom small weight initialization
            def small_initializer():
                return tf.keras.initializers.GlorotUniform(seed=42)
            
            # Re-initialize network layers with small weights
            for layer in net.layers:
                if hasattr(layer, 'kernel') and layer.kernel is not None:
                    # Scale existing weights by weight_scale
                    layer.kernel.assign(layer.kernel * weight_scale)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    # Scale biases as well
                    layer.bias.assign(layer.bias * weight_scale)
                    
            logger.info(f"Applied small weight initialization with scale: {weight_scale}")
            
        except Exception as e:
            logger.warning(f"Small weight initialization failed: {e}. Using default initialization.")
    
    def _train_with_adaptive_switching(self, config: Dict, callbacks: List) -> Any:
        """Train with adaptive optimizer switching based on plateau detection."""
        max_iterations = config['max_iterations']
        plateau_patience = 500  # Iterations to wait before considering plateau
        plateau_threshold = 1e-6  # Minimum improvement threshold
        
        iteration = 0
        plateau_counter = 0
        last_losses = []
        
        while iteration < max_iterations:
            # Train for small batches
            batch_iterations = min(100, max_iterations - iteration)
            
            # Train batch
            losshistory = self.model.train(
                iterations=batch_iterations,
                callbacks=callbacks,
                display_every=100
            )
            
            iteration += batch_iterations
            
            # Check for plateau
            if losshistory.loss_train:
                current_loss = losshistory.loss_train[-1]
                last_losses.append(current_loss)
                
                # Keep only recent losses for plateau detection
                if len(last_losses) > plateau_patience // 100:
                    last_losses = last_losses[-(plateau_patience // 100):]
                
                # Check if we've plateaued
                if len(last_losses) >= plateau_patience // 100:
                    improvement = max(last_losses) - min(last_losses)
                    if improvement < plateau_threshold:
                        plateau_counter += 1
                        if plateau_counter >= 3:  # Consistent plateau
                            logger.info(f"Detected plateau in Adam training at iteration {iteration}. "
                                       f"Switching to L-BFGS early.")
                            break
                    else:
                        plateau_counter = 0
        
        return losshistory
    
    def _create_stage_callbacks(self, stage_name: str) -> List:
        """Create appropriate callbacks for each training stage."""
        config = self.stage_configs[stage_name]
        callbacks = []
        
        # Early stopping
        early_stopping = dde.callbacks.EarlyStopping(
            min_delta=self.min_improvement_delta,
            patience=config.get('early_stopping_patience', 1000),
            monitor='loss_train'
        )
        callbacks.append(early_stopping)
        
        # Model checkpointing
        checkpoint_path = self.checkpoint_dir / f"{stage_name}_best.ckpt"
        model_checkpoint = dde.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            save_better_only=True,
            monitor='loss_train'
        )
        callbacks.append(model_checkpoint)
        
        # Variable tracking
        var_path = self.checkpoint_dir / f"{stage_name}_variables.dat"
        variable_tracker = dde.callbacks.VariableValue(
            [], 
            period=500,
            filename=str(var_path)
        )
        callbacks.append(variable_tracker)
        
        # Custom metrics tracking
        metrics_callback = MetricsTrackingCallback(self.metrics_tracker, stage_name)
        callbacks.append(metrics_callback)
        
        # Physics consistency checker
        if hasattr(self.loss_function, 'get_loss_statistics'):
            physics_checker = PhysicsConsistencyCallback(
                self.loss_function, 
                check_period=500
            )
            callbacks.append(physics_checker)
        
        return callbacks
    
    def _finalize_training(self) -> Dict[str, Any]:
        """Finalize training and prepare results."""
        # Load best model
        best_checkpoint = self._find_best_checkpoint()
        if best_checkpoint:
            self.model.restore(str(best_checkpoint))
            logger.info(f"Restored best model from {best_checkpoint}")
        
        # Gather final statistics
        final_results = {
            'success': True,
            'training_history': self.training_history,
            'final_metrics': self.metrics_tracker.get_summary(),
            'best_model_path': str(best_checkpoint) if best_checkpoint else None,
            'loss_evolution': self.loss_function.get_loss_statistics() if hasattr(self.loss_function, 'get_loss_statistics') else None,
            'stage_summaries': self._create_stage_summaries()
        }
        
        # Save training report
        self._save_training_report(final_results)
        
        return final_results
    
    def _find_best_checkpoint(self) -> Optional[Path]:
        """Find the checkpoint with the best performance."""
        checkpoints = list(self.checkpoint_dir.glob("*.ckpt*"))
        if not checkpoints:
            return None
        
        # Find checkpoint with lowest loss (simple heuristic)
        best_checkpoint = None
        best_loss = float('inf')
        
        for stage_name, history in self.training_history.items():
            if 'losshistory' in history and history['losshistory'].loss_train:
                stage_loss = min(history['losshistory'].loss_train)
                if stage_loss < best_loss:
                    best_loss = stage_loss
                    checkpoint_pattern = f"{stage_name}_best.ckpt"
                    matching_checkpoints = [cp for cp in checkpoints if checkpoint_pattern in cp.name]
                    if matching_checkpoints:
                        best_checkpoint = matching_checkpoints[0]
        
        return best_checkpoint
    
    def _create_stage_summaries(self) -> Dict[str, Dict]:
        """Create summary statistics for each training stage."""
        summaries = {}
        
        for stage_name, history in self.training_history.items():
            if 'losshistory' in history and history['losshistory'].loss_train:
                train_losses = history['losshistory'].loss_train
                summaries[stage_name] = {
                    'initial_loss': train_losses[0],
                    'final_loss': train_losses[-1],
                    'min_loss': min(train_losses),
                    'improvement': train_losses[0] - train_losses[-1],
                    'improvement_ratio': (train_losses[0] - train_losses[-1]) / train_losses[0],
                    'iterations': len(train_losses),
                    'converged': history['final_loss'] < self.min_improvement_delta * 10
                }
        
        return summaries
    
    def _save_training_report(self, results: Dict[str, Any]) -> None:
        """Save comprehensive training report."""
        report_path = self.checkpoint_dir / "training_report.json"
        
        # Convert non-serializable objects
        serializable_results = self._make_serializable(results)
        
        try:
            with open(report_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            logger.info(f"Training report saved to {report_path}")
        except Exception as e:
            logger.warning(f"Failed to save training report: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, tf.Tensor)):
            return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def _create_failure_result(self, error_message: str) -> Dict[str, Any]:
        """Create result dictionary for failed training."""
        return {
            'success': False,
            'error': error_message,
            'training_history': self.training_history,
            'partial_results': self.metrics_tracker.get_summary()
        }


class UVVisLeaveOneScanOutCV:
    """
    Leave-one-scan-out cross-validation for UV-Vis spectroscopy PINN.
    
    This class implements comprehensive cross-validation by systematically
    holding out each concentration level and evaluating model performance.
    """
    
    def __init__(self, 
                 data_processor: UVVisDataProcessor,
                 model_factory: Callable,
                 loss_factory: Callable,
                 cv_results_dir: str = "cv_results"):
        """
        Initialize cross-validation strategy.
        
        Args:
            data_processor: Data processor with loaded UV-Vis data
            model_factory: Factory function to create models
            loss_factory: Factory function to create loss functions
            cv_results_dir: Directory for CV results
        """
        self.data_processor = data_processor
        self.model_factory = model_factory
        self.loss_factory = loss_factory
        self.cv_results_dir = Path(cv_results_dir)
        
        # Create results directory
        self.cv_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract concentration levels for CV
        self.concentration_levels = self.data_processor.concentrations_nonzero
        
        logger.info(f"Initialized CV with {len(self.concentration_levels)} concentration levels: "
                   f"{self.concentration_levels}")
    
    def execute_cross_validation(self, parallel: bool = False) -> Dict[str, Any]:
        """
        Execute complete leave-one-scan-out cross-validation.
        
        Args:
            parallel: Whether to execute CV folds in parallel
            
        Returns:
            Comprehensive CV results
        """
        logger.info("Starting leave-one-scan-out cross-validation...")
        start_time = time.time()
        
        if parallel:
            cv_results = self._execute_parallel_cv()
        else:
            cv_results = self._execute_sequential_cv()
        
        # Aggregate results
        aggregated_results = self._aggregate_cv_results(cv_results)
        
        # Save comprehensive report
        self._save_cv_report(aggregated_results)
        
        total_time = time.time() - start_time
        logger.info(f"Cross-validation completed in {total_time:.2f} seconds")
        
        return aggregated_results
    
    def _execute_sequential_cv(self) -> List[Dict]:
        """Execute CV folds sequentially."""
        cv_results = []
        
        for fold_idx, held_out_conc in enumerate(self.concentration_levels):
            logger.info(f"CV Fold {fold_idx + 1}/{len(self.concentration_levels)}: "
                       f"Holding out concentration {held_out_conc} µg/L")
            
            fold_result = self._execute_single_fold(fold_idx, held_out_conc)
            cv_results.append(fold_result)
        
        return cv_results
    
    def _execute_parallel_cv(self) -> List[Dict]:
        """Execute CV folds in parallel."""
        logger.info("Executing CV folds in parallel...")
        
        with ThreadPoolExecutor(max_workers=min(4, len(self.concentration_levels))) as executor:
            futures = []
            
            for fold_idx, held_out_conc in enumerate(self.concentration_levels):
                future = executor.submit(self._execute_single_fold, fold_idx, held_out_conc)
                futures.append(future)
            
            cv_results = [future.result() for future in futures]
        
        return cv_results
    
    def _execute_single_fold(self, fold_idx: int, held_out_conc: float) -> Dict[str, Any]:
        """
        Execute a single cross-validation fold.
        
        Args:
            fold_idx: Fold index
            held_out_conc: Concentration to hold out
            
        Returns:
            Single fold results
        """
        try:
            # Create train/test split
            train_data = self._create_train_data(held_out_conc)
            test_data = self._create_test_data(held_out_conc)
            
            # Create model and loss function for this fold
            model = self.model_factory(train_data)
            loss_fn = self.loss_factory()
            
            # Create training strategy
            fold_checkpoint_dir = self.cv_results_dir / f"fold_{fold_idx}"
            training_strategy = UVVisTrainingStrategy(
                model=model,
                loss_function=loss_fn,
                checkpoint_dir=str(fold_checkpoint_dir)
            )
            
            # Train model
            training_results = training_strategy.execute_full_training()
            
            # Evaluate on test data
            test_metrics = self._evaluate_fold(model, test_data)
            
            # Create fold result
            fold_result = {
                'fold_idx': fold_idx,
                'held_out_concentration': held_out_conc,
                'training_results': training_results,
                'test_metrics': test_metrics,
                'train_size': len(train_data[0]),
                'test_size': len(test_data[0]),
                'success': training_results['success']
            }
            
            logger.info(f"Fold {fold_idx} completed. Test loss: {test_metrics.get('loss', 'N/A')}")
            
            return fold_result
            
        except Exception as e:
            logger.error(f"Fold {fold_idx} failed: {e}")
            return {
                'fold_idx': fold_idx,
                'held_out_concentration': held_out_conc,
                'success': False,
                'error': str(e)
            }
    
    def _create_train_data(self, held_out_conc: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create training data excluding held-out concentration."""
        # Get full training data
        X_full, y_full = self.data_processor.create_training_data()
        
        # Find indices to exclude
        concentrations_norm = self.data_processor.normalize_inputs()['concentrations_norm']
        held_out_conc_norm = held_out_conc / self.data_processor.concentration_norm_params['scale']
        
        # Create mask for training data (exclude held-out concentration)
        exclude_indices = []
        for i, (wl, conc) in enumerate(X_full):
            if np.abs(conc - held_out_conc_norm) < 1e-6:
                exclude_indices.append(i)
        
        # Create training data
        train_indices = [i for i in range(len(X_full)) if i not in exclude_indices]
        X_train = X_full[train_indices]
        y_train = y_full[train_indices]
        
        return X_train, y_train
    
    def _create_test_data(self, held_out_conc: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create test data for held-out concentration."""
        # Get full training data
        X_full, y_full = self.data_processor.create_training_data()
        
        # Find indices for test data
        concentrations_norm = self.data_processor.normalize_inputs()['concentrations_norm']
        held_out_conc_norm = held_out_conc / self.data_processor.concentration_norm_params['scale']
        
        # Create mask for test data (only held-out concentration)
        test_indices = []
        for i, (wl, conc) in enumerate(X_full):
            if np.abs(conc - held_out_conc_norm) < 1e-6:
                test_indices.append(i)
        
        # Create test data
        X_test = X_full[test_indices]
        y_test = y_full[test_indices]
        
        return X_test, y_test
    
    def _evaluate_fold(self, model: dde.Model, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        X_test, y_test = test_data
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Compute metrics
            mse = np.mean((y_pred - y_test) ** 2)
            mae = np.mean(np.abs(y_pred - y_test))
            rmse = np.sqrt(mse)
            
            # R-squared
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))
            
            return {
                'loss': float(mse),
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'predictions': y_pred.tolist(),
                'targets': y_test.tolist()
            }
            
        except Exception as e:
            logger.error(f"Fold evaluation failed: {e}")
            return {'error': str(e)}
    
    def _aggregate_cv_results(self, cv_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across all CV folds."""
        successful_folds = [r for r in cv_results if r.get('success', False)]
        
        if not successful_folds:
            return {
                'success': False,
                'error': 'No successful CV folds',
                'raw_results': cv_results
            }
        
        # Aggregate metrics
        metrics = ['mse', 'mae', 'rmse', 'r2']
        aggregated_metrics = {}
        
        for metric in metrics:
            values = [fold['test_metrics'][metric] for fold in successful_folds 
                     if metric in fold.get('test_metrics', {})]
            
            if values:
                aggregated_metrics[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'values': values
                }
        
        return {
            'success': True,
            'n_successful_folds': len(successful_folds),
            'n_total_folds': len(cv_results),
            'aggregated_metrics': aggregated_metrics,
            'fold_details': successful_folds,
            'failed_folds': [r for r in cv_results if not r.get('success', False)],
            'concentration_performance': self._analyze_concentration_performance(successful_folds)
        }
    
    def _analyze_concentration_performance(self, successful_folds: List[Dict]) -> Dict:
        """Analyze performance vs concentration levels."""
        concentration_analysis = {}
        
        for fold in successful_folds:
            conc = fold['held_out_concentration']
            metrics = fold['test_metrics']
            
            concentration_analysis[conc] = {
                'mse': metrics.get('mse', float('nan')),
                'mae': metrics.get('mae', float('nan')),
                'r2': metrics.get('r2', float('nan'))
            }
        
        return concentration_analysis
    
    def _save_cv_report(self, results: Dict[str, Any]) -> None:
        """Save comprehensive CV report."""
        report_path = self.cv_results_dir / "cv_report.json"
        
        try:
            # Make serializable
            serializable_results = self._make_serializable(results)
            
            with open(report_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
                
            logger.info(f"CV report saved to {report_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save CV report: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, tf.Tensor)):
            return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj


class TrainingMetricsTracker:
    """Track comprehensive training metrics across all stages."""
    
    def __init__(self):
        self.metrics_history = {}
        self.stage_metrics = {}
        self.convergence_events = []
    
    def update_metrics(self, stage: str, iteration: int, metrics: Dict[str, float]) -> None:
        """Update metrics for current training stage."""
        if stage not in self.metrics_history:
            self.metrics_history[stage] = {}
        
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history[stage]:
                self.metrics_history[stage][metric_name] = []
            self.metrics_history[stage][metric_name].append((iteration, value))
    
    def record_convergence_event(self, stage: str, iteration: int, event_type: str, details: Dict) -> None:
        """Record significant convergence events."""
        self.convergence_events.append({
            'stage': stage,
            'iteration': iteration,
            'event_type': event_type,
            'details': details,
            'timestamp': time.time()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of training metrics."""
        return {
            'metrics_history': self.metrics_history,
            'convergence_events': self.convergence_events,
            'stage_summaries': self.stage_metrics
        }


class MetricsTrackingCallback(dde.callbacks.Callback):
    """Custom callback for tracking training metrics."""
    
    def __init__(self, metrics_tracker: TrainingMetricsTracker, stage_name: str):
        super().__init__()
        self.metrics_tracker = metrics_tracker
        self.stage_name = stage_name
        self.last_loss = float('inf')
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Track metrics at the end of each epoch."""
        if logs:
            # Extract relevant metrics
            metrics = {
                'train_loss': logs.get('loss', 0.0),
                'test_loss': logs.get('val_loss', logs.get('test_loss', 0.0))
            }
            
            # Update tracker
            self.metrics_tracker.update_metrics(self.stage_name, epoch, metrics)
            
            # Check for significant improvements
            current_loss = metrics['train_loss']
            if current_loss < self.last_loss * 0.9:  # 10% improvement
                self.metrics_tracker.record_convergence_event(
                    self.stage_name, 
                    epoch, 
                    'significant_improvement',
                    {'old_loss': self.last_loss, 'new_loss': current_loss}
                )
            
            self.last_loss = current_loss


class PhysicsConsistencyCallback(dde.callbacks.Callback):
    """Custom callback to monitor physics constraint satisfaction."""
    
    def __init__(self, loss_function: UVVisLossFunction, check_period: int = 500):
        super().__init__()
        self.loss_function = loss_function
        self.check_period = check_period
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Check physics consistency periodically."""
        if epoch % self.check_period == 0:
            try:
                # Get loss statistics if available
                if hasattr(self.loss_function, 'get_loss_statistics'):
                    loss_stats = self.loss_function.get_loss_statistics()
                    
                    # Log physics consistency metrics
                    if 'physics' in loss_stats:
                        physics_loss = loss_stats['physics']['current']
                        logger.info(f"Physics consistency check at epoch {epoch}: "
                                   f"Physics loss = {physics_loss:.6f}")
                        
                        # Warn if physics loss is increasing
                        if 'trend' in loss_stats['physics'] and loss_stats['physics']['trend'] == 'increasing':
                            logger.warning(f"Physics loss is increasing at epoch {epoch}")
                            
            except Exception as e:
                logger.debug(f"Physics consistency check failed at epoch {epoch}: {e}")


class LearningRateScheduler(dde.callbacks.Callback):
    """Custom learning rate scheduler."""
    
    def __init__(self, initial_lr: float, decay_factor: float = 0.9, decay_steps: int = 1000):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Update learning rate according to schedule."""
        if epoch > 0 and epoch % self.decay_steps == 0:
            new_lr = self.initial_lr * (self.decay_factor ** (epoch // self.decay_steps))
            
            # Update optimizer learning rate (implementation depends on backend)
            try:
                if hasattr(self.model, 'optimizer') and hasattr(self.model.optimizer, 'learning_rate'):
                    self.model.optimizer.learning_rate.assign(new_lr)
                    logger.info(f"Learning rate updated to {new_lr:.6f} at epoch {epoch}")
            except Exception as e:
                logger.debug(f"Learning rate update failed: {e}")


def create_training_pipeline(data_processor: UVVisDataProcessor,
                           model_factory: Callable,
                           loss_factory: Callable,
                           enable_cv: bool = True,
                           cv_parallel: bool = False) -> Dict[str, Any]:
    """
    Create and execute complete training pipeline.
    
    Args:
        data_processor: Data processor with loaded UV-Vis data
        model_factory: Factory function to create models
        loss_factory: Factory function to create loss functions
        enable_cv: Whether to perform cross-validation
        cv_parallel: Whether to run CV in parallel
        
    Returns:
        Complete training results
    """
    logger.info("Creating complete UV-Vis PINN training pipeline...")
    
    pipeline_results = {
        'data_summary': data_processor.get_data_summary(),
        'training_results': None,
        'cv_results': None,
        'success': False
    }
    
    try:
        if enable_cv:
            # Run cross-validation
            cv_strategy = UVVisLeaveOneScanOutCV(
                data_processor=data_processor,
                model_factory=model_factory,
                loss_factory=loss_factory
            )
            
            pipeline_results['cv_results'] = cv_strategy.execute_cross_validation(parallel=cv_parallel)
        
        else:
            # Single training run on full dataset
            X_train, y_train = data_processor.create_training_data()
            model = model_factory((X_train, y_train))
            loss_fn = loss_factory()
            
            training_strategy = UVVisTrainingStrategy(
                model=model,
                loss_function=loss_fn
            )
            
            pipeline_results['training_results'] = training_strategy.execute_full_training()
        
        pipeline_results['success'] = True
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        pipeline_results['error'] = str(e)
    
    return pipeline_results


if __name__ == "__main__":
    # Example usage
    logger.info("Testing UV-Vis PINN training implementation...")
    
    # This would typically be called with actual data processor, model factory, and loss factory
    print("Training pipeline implementation completed successfully!")
    print("Use create_training_pipeline() to execute full training with your data.")
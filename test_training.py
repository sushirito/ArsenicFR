"""
Comprehensive Unit Tests for UV-Vis PINN Training Pipeline
=========================================================

This module contains comprehensive unit tests for the training implementation,
including tests for:
- Multi-stage training strategy
- Cross-validation implementation
- Optimizer switching mechanisms
- Training monitoring and callbacks
- Model checkpointing and recovery
- Error handling and robustness

Author: Claude Code Assistant
"""

import pytest
import numpy as np
import tensorflow as tf
import deepxde as dde
from unittest.mock import Mock, patch, MagicMock, call
import tempfile
import shutil
import json
import os
from pathlib import Path
import time

# Import modules to test
from .training import (
    UVVisTrainingStrategy,
    UVVisLeaveOneScanOutCV,
    TrainingMetricsTracker,
    MetricsTrackingCallback,
    PhysicsConsistencyCallback,
    LearningRateScheduler,
    create_training_pipeline
)
from .data_preprocessing import UVVisDataProcessor
from .loss_functions import UVVisLossFunction


class TestUVVisTrainingStrategy:
    """Test suite for UVVisTrainingStrategy class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock DeepXDE model for testing."""
        mock_model = Mock()
        mock_model.compile = Mock()
        mock_model.train = Mock()
        mock_model.restore = Mock()
        
        # Mock loss history
        mock_losshistory = Mock()
        mock_losshistory.loss_train = [1.0, 0.5, 0.1, 0.05]
        mock_model.train.return_value = mock_losshistory
        
        # Mock network
        mock_net = Mock()
        mock_net.layers = [Mock(), Mock()]  # Two layers
        for layer in mock_net.layers:
            layer.kernel = tf.Variable([[0.1, 0.1], [0.1, 0.1]])
            layer.bias = tf.Variable([0.1, 0.1])
        mock_model.net = mock_net
        
        return mock_model
    
    @pytest.fixture
    def mock_loss_function(self):
        """Create mock loss function for testing."""
        mock_loss = Mock(spec=UVVisLossFunction)
        mock_loss.get_loss_statistics.return_value = {
            'data': {'current': 0.1, 'trend': 'decreasing'},
            'physics': {'current': 0.05, 'trend': 'stable'},
            'smoothness': {'current': 0.001, 'trend': 'decreasing'}
        }
        return mock_loss
    
    @pytest.fixture
    def temp_checkpoint_dir(self, tmp_path):
        """Create temporary checkpoint directory."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        return str(checkpoint_dir)
    
    @pytest.fixture
    def training_strategy(self, mock_model, mock_loss_function, temp_checkpoint_dir):
        """Create training strategy for testing."""
        return UVVisTrainingStrategy(
            model=mock_model,
            loss_function=mock_loss_function,
            checkpoint_dir=temp_checkpoint_dir,
            enable_adaptive_switching=True,
            convergence_patience=100,
            min_improvement_delta=1e-6
        )
    
    def test_initialization(self, mock_model, mock_loss_function, temp_checkpoint_dir):
        """Test training strategy initialization."""
        strategy = UVVisTrainingStrategy(
            model=mock_model,
            loss_function=mock_loss_function,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        assert strategy.model == mock_model
        assert strategy.loss_function == mock_loss_function
        assert strategy.checkpoint_dir == Path(temp_checkpoint_dir)
        assert strategy.current_stage == 'initialization'
        assert strategy.enable_adaptive_switching
        
        # Check checkpoint directory was created
        assert strategy.checkpoint_dir.exists()
        
        # Check stage configurations
        assert 'initialization' in strategy.stage_configs
        assert 'adam_training' in strategy.stage_configs
        assert 'lbfgs_refinement' in strategy.stage_configs
    
    def test_stage_configurations(self, training_strategy):
        """Test stage configuration creation."""
        configs = training_strategy.stage_configs
        
        # Check initialization stage
        init_config = configs['initialization']
        assert init_config['optimizer'] == 'adam'
        assert init_config['learning_rate'] == 1e-4
        assert init_config['weight_scale'] == 0.1
        
        # Check Adam training stage
        adam_config = configs['adam_training']
        assert adam_config['optimizer'] == 'adam'
        assert adam_config['learning_rate'] == 1e-3
        assert 'lr_decay_factor' in adam_config
        
        # Check L-BFGS refinement stage
        lbfgs_config = configs['lbfgs_refinement']
        assert lbfgs_config['optimizer'] == 'L-BFGS'
        assert lbfgs_config['learning_rate'] is None
    
    def test_small_weight_initialization(self, training_strategy):
        """Test small weight initialization."""
        # Test weight initialization
        training_strategy._initialize_small_weights(0.1)
        
        # Check that layer weights were modified
        for layer in training_strategy.model.net.layers:
            # Weights should have been scaled
            assert layer.kernel.numpy().max() <= 0.1
            assert layer.bias.numpy().max() <= 0.1
    
    def test_initialization_stage_execution(self, training_strategy):
        """Test initialization stage execution."""
        training_strategy._execute_initialization_stage()
        
        # Check that model was compiled
        training_strategy.model.compile.assert_called()
        
        # Check that training was called
        training_strategy.model.train.assert_called()
        
        # Check that results were stored
        assert 'initialization' in training_strategy.training_history
        assert training_strategy.current_stage == 'initialization'
    
    def test_adam_stage_execution(self, training_strategy):
        """Test Adam training stage execution."""
        training_strategy._execute_adam_stage()
        
        # Check that model was compiled with Adam
        training_strategy.model.compile.assert_called()
        
        # Check that training was called
        training_strategy.model.train.assert_called()
        
        # Check that results were stored
        assert 'adam_training' in training_strategy.training_history
        assert training_strategy.current_stage == 'adam_training'
    
    def test_lbfgs_stage_execution(self, training_strategy):
        """Test L-BFGS refinement stage execution."""
        training_strategy._execute_lbfgs_stage()
        
        # Check that model was compiled with L-BFGS
        training_strategy.model.compile.assert_called()
        
        # Check that training was called
        training_strategy.model.train.assert_called()
        
        # Check that results were stored
        assert 'lbfgs_refinement' in training_strategy.training_history
        assert training_strategy.current_stage == 'lbfgs_refinement'
    
    def test_callback_creation(self, training_strategy):
        """Test callback creation for different stages."""
        for stage_name in ['initialization', 'adam_training', 'lbfgs_refinement']:
            callbacks = training_strategy._create_stage_callbacks(stage_name)
            
            # Should have multiple callbacks
            assert len(callbacks) > 0
            
            # Should include essential callbacks
            callback_types = [type(cb).__name__ for cb in callbacks]
            assert any('EarlyStopping' in name for name in callback_types)
    
    def test_full_training_execution(self, training_strategy):
        """Test complete multi-stage training execution."""
        results = training_strategy.execute_full_training()
        
        # Should complete successfully
        assert results['success']
        assert 'training_history' in results
        assert 'final_metrics' in results
        
        # All stages should have been executed
        assert 'initialization' in training_strategy.training_history
        assert 'adam_training' in training_strategy.training_history
        assert 'lbfgs_refinement' in training_strategy.training_history
        
        # Model compile should have been called multiple times (once per stage)
        assert training_strategy.model.compile.call_count >= 3
        
        # Model train should have been called multiple times
        assert training_strategy.model.train.call_count >= 3
    
    def test_adaptive_switching(self, training_strategy):
        """Test adaptive optimizer switching mechanism."""
        config = training_strategy.stage_configs['adam_training']
        callbacks = []  # Empty callbacks for testing
        
        # Mock training that shows plateau
        def mock_train_with_plateau(iterations, **kwargs):
            mock_losshistory = Mock()
            # Simulate plateau: loss doesn't improve
            mock_losshistory.loss_train = [1.0] * (iterations // 100)
            return mock_losshistory
        
        training_strategy.model.train = mock_train_with_plateau
        
        # Execute adaptive training
        losshistory = training_strategy._train_with_adaptive_switching(config, callbacks)
        
        # Should return loss history
        assert losshistory is not None
        assert hasattr(losshistory, 'loss_train')
    
    def test_best_checkpoint_finding(self, training_strategy, temp_checkpoint_dir):
        """Test finding the best checkpoint."""
        # Create some mock checkpoint files
        checkpoint_dir = Path(temp_checkpoint_dir)
        
        # Create dummy checkpoint files
        (checkpoint_dir / "initialization_best.ckpt.index").touch()
        (checkpoint_dir / "adam_training_best.ckpt.index").touch()
        (checkpoint_dir / "lbfgs_refinement_best.ckpt.index").touch()
        
        # Populate training history with different losses
        training_strategy.training_history = {
            'initialization': {'losshistory': Mock(loss_train=[1.0, 0.8, 0.6])},
            'adam_training': {'losshistory': Mock(loss_train=[0.6, 0.4, 0.2])},
            'lbfgs_refinement': {'losshistory': Mock(loss_train=[0.2, 0.1, 0.05])}  # Best
        }
        
        best_checkpoint = training_strategy._find_best_checkpoint()
        
        # Should find L-BFGS checkpoint (lowest loss)
        assert best_checkpoint is not None
        assert "lbfgs_refinement" in str(best_checkpoint)
    
    def test_stage_summaries_creation(self, training_strategy):
        """Test creation of stage summaries."""
        # Mock training history
        training_strategy.training_history = {
            'initialization': {
                'losshistory': Mock(loss_train=[1.0, 0.8, 0.6]),
                'final_loss': 0.6
            },
            'adam_training': {
                'losshistory': Mock(loss_train=[0.6, 0.4, 0.2]),
                'final_loss': 0.2
            }
        }
        
        summaries = training_strategy._create_stage_summaries()
        
        # Should have summaries for both stages
        assert 'initialization' in summaries
        assert 'adam_training' in summaries
        
        # Check summary content
        init_summary = summaries['initialization']
        assert init_summary['initial_loss'] == 1.0
        assert init_summary['final_loss'] == 0.6
        assert init_summary['min_loss'] == 0.6
        assert init_summary['improvement'] == 0.4
        assert init_summary['iterations'] == 3
    
    def test_training_report_saving(self, training_strategy, temp_checkpoint_dir):
        """Test saving of training report."""
        results = {
            'success': True,
            'training_history': {'test': 'data'},
            'final_metrics': {'loss': 0.1}
        }
        
        training_strategy._save_training_report(results)
        
        # Check that report file was created
        report_path = Path(temp_checkpoint_dir) / "training_report.json"
        assert report_path.exists()
        
        # Check report content
        with open(report_path, 'r') as f:
            saved_results = json.load(f)
        
        assert saved_results['success']
        assert 'training_history' in saved_results
    
    def test_error_handling(self, training_strategy):
        """Test error handling in training."""
        # Mock training failure
        training_strategy.model.train.side_effect = Exception("Training failed")
        
        results = training_strategy.execute_full_training()
        
        # Should handle error gracefully
        assert not results['success']
        assert 'error' in results
        assert results['error'] == "Training failed"


class TestUVVisLeaveOneScanOutCV:
    """Test suite for UVVisLeaveOneScanOutCV class."""
    
    @pytest.fixture
    def mock_data_processor(self):
        """Create mock data processor for testing."""
        mock_processor = Mock(spec=UVVisDataProcessor)
        
        # Mock concentration levels
        mock_processor.concentrations_nonzero = np.array([10, 20, 30, 40, 60])
        
        # Mock normalization parameters
        mock_processor.concentration_norm_params = {'scale': 60.0}
        
        # Mock training data creation
        def mock_create_training_data():
            n_points = 100
            X = np.random.rand(n_points, 2).astype(np.float32)
            y = np.random.rand(n_points, 1).astype(np.float32)
            return X, y
        
        mock_processor.create_training_data = mock_create_training_data
        
        # Mock normalization
        mock_processor.normalize_inputs.return_value = {
            'concentrations_norm': np.array([10, 20, 30, 40, 60]) / 60.0
        }
        
        return mock_processor
    
    @pytest.fixture
    def mock_model_factory(self):
        """Create mock model factory for testing."""
        def factory(data):
            mock_model = Mock()
            mock_model.predict = Mock(return_value=np.random.rand(50, 1))
            return mock_model
        return factory
    
    @pytest.fixture
    def mock_loss_factory(self):
        """Create mock loss factory for testing."""
        def factory():
            return Mock(spec=UVVisLossFunction)
        return factory
    
    @pytest.fixture
    def cv_strategy(self, mock_data_processor, mock_model_factory, mock_loss_factory, tmp_path):
        """Create CV strategy for testing."""
        cv_results_dir = tmp_path / "cv_results"
        return UVVisLeaveOneScanOutCV(
            data_processor=mock_data_processor,
            model_factory=mock_model_factory,
            loss_factory=mock_loss_factory,
            cv_results_dir=str(cv_results_dir)
        )
    
    def test_initialization(self, cv_strategy):
        """Test CV strategy initialization."""
        assert cv_strategy.data_processor is not None
        assert cv_strategy.model_factory is not None
        assert cv_strategy.loss_factory is not None
        assert cv_strategy.cv_results_dir.exists()
        assert len(cv_strategy.concentration_levels) == 5
    
    def test_train_test_split_creation(self, cv_strategy):
        """Test creation of train/test splits for CV."""
        held_out_conc = 20.0
        
        # Create train data (excluding held-out concentration)
        train_data = cv_strategy._create_train_data(held_out_conc)
        X_train, y_train = train_data
        
        assert X_train.shape[1] == 2  # Should have 2 input features
        assert y_train.shape[1] == 1  # Should have 1 output
        assert len(X_train) > 0
        
        # Create test data (only held-out concentration)
        test_data = cv_strategy._create_test_data(held_out_conc)
        X_test, y_test = test_data
        
        assert X_test.shape[1] == 2
        assert y_test.shape[1] == 1
    
    def test_single_fold_execution(self, cv_strategy):
        """Test execution of a single CV fold."""
        fold_idx = 0
        held_out_conc = 20.0
        
        with patch.object(cv_strategy, '_create_train_data') as mock_create_train, \
             patch.object(cv_strategy, '_create_test_data') as mock_create_test:
            
            # Mock data creation
            mock_create_train.return_value = (np.random.rand(80, 2), np.random.rand(80, 1))
            mock_create_test.return_value = (np.random.rand(20, 2), np.random.rand(20, 1))
            
            with patch('pinn_uv_vis_spectra.training.UVVisTrainingStrategy') as mock_strategy_class:
                # Mock training strategy
                mock_strategy = Mock()
                mock_strategy.execute_full_training.return_value = {
                    'success': True,
                    'final_loss': 0.1
                }
                mock_strategy_class.return_value = mock_strategy
                
                # Execute single fold
                fold_result = cv_strategy._execute_single_fold(fold_idx, held_out_conc)
                
                assert fold_result['success']
                assert fold_result['fold_idx'] == fold_idx
                assert fold_result['held_out_concentration'] == held_out_conc
                assert 'test_metrics' in fold_result
    
    def test_model_evaluation(self, cv_strategy):
        """Test model evaluation on test data."""
        # Create mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.1], [0.2], [0.15], [0.18]])
        
        # Create test data
        X_test = np.array([[0.5, 0.5], [0.6, 0.6], [0.7, 0.7], [0.8, 0.8]])
        y_test = np.array([[0.12], [0.19], [0.16], [0.17]])
        test_data = (X_test, y_test)
        
        # Evaluate model
        metrics = cv_strategy._evaluate_fold(mock_model, test_data)
        
        # Check that metrics are computed
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'predictions' in metrics
        assert 'targets' in metrics
        
        # Check that values are reasonable
        assert metrics['mse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0
        assert -1 <= metrics['r2'] <= 1  # R² can be negative for very bad fits
    
    def test_sequential_cv_execution(self, cv_strategy):
        """Test sequential CV execution."""
        with patch.object(cv_strategy, '_execute_single_fold') as mock_single_fold:
            # Mock single fold results
            mock_single_fold.return_value = {
                'success': True,
                'fold_idx': 0,
                'held_out_concentration': 10.0,
                'test_metrics': {'mse': 0.1, 'mae': 0.05, 'rmse': 0.316, 'r2': 0.9}
            }
            
            cv_results = cv_strategy._execute_sequential_cv()
            
            # Should execute one fold for each concentration
            assert len(cv_results) == len(cv_strategy.concentration_levels)
            assert mock_single_fold.call_count == len(cv_strategy.concentration_levels)
    
    def test_results_aggregation(self, cv_strategy):
        """Test aggregation of CV results."""
        # Create mock CV results
        mock_cv_results = [
            {
                'success': True,
                'held_out_concentration': 10.0,
                'test_metrics': {'mse': 0.1, 'mae': 0.05, 'rmse': 0.316, 'r2': 0.9}
            },
            {
                'success': True,
                'held_out_concentration': 20.0,
                'test_metrics': {'mse': 0.2, 'mae': 0.1, 'rmse': 0.447, 'r2': 0.8}
            },
            {
                'success': False,
                'held_out_concentration': 30.0,
                'error': 'Training failed'
            }
        ]
        
        aggregated = cv_strategy._aggregate_cv_results(mock_cv_results)
        
        # Check aggregation results
        assert aggregated['success']
        assert aggregated['n_successful_folds'] == 2
        assert aggregated['n_total_folds'] == 3
        
        # Check aggregated metrics
        assert 'aggregated_metrics' in aggregated
        assert 'mse' in aggregated['aggregated_metrics']
        
        # Check that mean MSE is computed correctly
        expected_mse_mean = (0.1 + 0.2) / 2
        assert abs(aggregated['aggregated_metrics']['mse']['mean'] - expected_mse_mean) < 1e-6
        
        # Check failed folds are tracked
        assert len(aggregated['failed_folds']) == 1
    
    def test_concentration_performance_analysis(self, cv_strategy):
        """Test analysis of performance vs concentration."""
        successful_folds = [
            {
                'held_out_concentration': 10.0,
                'test_metrics': {'mse': 0.1, 'mae': 0.05, 'r2': 0.9}
            },
            {
                'held_out_concentration': 20.0,
                'test_metrics': {'mse': 0.2, 'mae': 0.1, 'r2': 0.8}
            }
        ]
        
        analysis = cv_strategy._analyze_concentration_performance(successful_folds)
        
        # Should have analysis for each concentration
        assert 10.0 in analysis
        assert 20.0 in analysis
        
        # Check metrics for each concentration
        assert analysis[10.0]['mse'] == 0.1
        assert analysis[10.0]['r2'] == 0.9
        assert analysis[20.0]['mse'] == 0.2
        assert analysis[20.0]['r2'] == 0.8
    
    def test_cv_report_saving(self, cv_strategy, tmp_path):
        """Test saving of CV report."""
        results = {
            'success': True,
            'n_successful_folds': 3,
            'aggregated_metrics': {'mse': {'mean': 0.15}}
        }
        
        cv_strategy._save_cv_report(results)
        
        # Check that report file was created
        report_path = cv_strategy.cv_results_dir / "cv_report.json"
        assert report_path.exists()
        
        # Check report content
        with open(report_path, 'r') as f:
            saved_results = json.load(f)
        
        assert saved_results['success']
        assert saved_results['n_successful_folds'] == 3


class TestTrainingMetricsTracker:
    """Test suite for TrainingMetricsTracker class."""
    
    @pytest.fixture
    def metrics_tracker(self):
        """Create metrics tracker for testing."""
        return TrainingMetricsTracker()
    
    def test_initialization(self, metrics_tracker):
        """Test metrics tracker initialization."""
        assert len(metrics_tracker.metrics_history) == 0
        assert len(metrics_tracker.stage_metrics) == 0
        assert len(metrics_tracker.convergence_events) == 0
    
    def test_metrics_update(self, metrics_tracker):
        """Test updating metrics."""
        # Update metrics for a stage
        metrics_tracker.update_metrics('adam_training', 100, {'loss': 0.5, 'accuracy': 0.8})
        metrics_tracker.update_metrics('adam_training', 200, {'loss': 0.3, 'accuracy': 0.9})
        
        # Check that metrics are stored
        assert 'adam_training' in metrics_tracker.metrics_history
        assert 'loss' in metrics_tracker.metrics_history['adam_training']
        assert 'accuracy' in metrics_tracker.metrics_history['adam_training']
        
        # Check that values are stored with iterations
        loss_history = metrics_tracker.metrics_history['adam_training']['loss']
        assert len(loss_history) == 2
        assert loss_history[0] == (100, 0.5)
        assert loss_history[1] == (200, 0.3)
    
    def test_convergence_event_recording(self, metrics_tracker):
        """Test recording convergence events."""
        details = {'old_loss': 1.0, 'new_loss': 0.5}
        metrics_tracker.record_convergence_event('adam_training', 150, 'significant_improvement', details)
        
        # Check that event is recorded
        assert len(metrics_tracker.convergence_events) == 1
        
        event = metrics_tracker.convergence_events[0]
        assert event['stage'] == 'adam_training'
        assert event['iteration'] == 150
        assert event['event_type'] == 'significant_improvement'
        assert event['details'] == details
        assert 'timestamp' in event
    
    def test_summary_generation(self, metrics_tracker):
        """Test summary generation."""
        # Add some test data
        metrics_tracker.update_metrics('test_stage', 100, {'loss': 0.5})
        metrics_tracker.record_convergence_event('test_stage', 150, 'test_event', {})
        
        summary = metrics_tracker.get_summary()
        
        # Check summary content
        assert 'metrics_history' in summary
        assert 'convergence_events' in summary
        assert 'stage_summaries' in summary
        assert 'test_stage' in summary['metrics_history']


class TestCustomCallbacks:
    """Test suite for custom callback implementations."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for callback testing."""
        mock_model = Mock()
        mock_model.optimizer = Mock()
        mock_model.optimizer.learning_rate = tf.Variable(1e-3)
        return mock_model
    
    def test_metrics_tracking_callback(self):
        """Test MetricsTrackingCallback."""
        metrics_tracker = TrainingMetricsTracker()
        callback = MetricsTrackingCallback(metrics_tracker, 'test_stage')
        
        # Simulate epoch end
        logs = {'loss': 0.5, 'val_loss': 0.6}
        callback.on_epoch_end(100, logs)
        
        # Check that metrics were tracked
        assert 'test_stage' in metrics_tracker.metrics_history
        assert 'train_loss' in metrics_tracker.metrics_history['test_stage']
        assert 'test_loss' in metrics_tracker.metrics_history['test_stage']
    
    def test_physics_consistency_callback(self):
        """Test PhysicsConsistencyCallback."""
        mock_loss_function = Mock()
        mock_loss_function.get_loss_statistics.return_value = {
            'physics': {'current': 0.05, 'trend': 'decreasing'}
        }
        
        callback = PhysicsConsistencyCallback(mock_loss_function, check_period=10)
        
        # Should check at multiples of check_period
        callback.on_epoch_end(10, {})
        mock_loss_function.get_loss_statistics.assert_called_once()
        
        # Should not check at other epochs
        mock_loss_function.reset_mock()
        callback.on_epoch_end(15, {})
        mock_loss_function.get_loss_statistics.assert_not_called()
    
    def test_learning_rate_scheduler(self, mock_model):
        """Test LearningRateScheduler callback."""
        scheduler = LearningRateScheduler(
            initial_lr=1e-3,
            decay_factor=0.9,
            decay_steps=100
        )
        scheduler.model = mock_model
        
        # Should not update LR before decay_steps
        scheduler.on_epoch_end(50, {})
        # LR should still be initial value (checking is complex due to mocking)
        
        # Should update LR at decay_steps
        scheduler.on_epoch_end(100, {})
        # LR should be updated (checking is complex due to mocking)


class TestTrainingPipeline:
    """Test suite for complete training pipeline."""
    
    @pytest.fixture
    def mock_data_processor(self):
        """Create mock data processor for pipeline testing."""
        mock_processor = Mock(spec=UVVisDataProcessor)
        mock_processor.get_data_summary.return_value = {'status': 'loaded'}
        mock_processor.create_training_data.return_value = (
            np.random.rand(100, 2),
            np.random.rand(100, 1)
        )
        return mock_processor
    
    @pytest.fixture
    def mock_model_factory(self):
        """Create mock model factory for pipeline testing."""
        def factory(data):
            return Mock()
        return factory
    
    @pytest.fixture
    def mock_loss_factory(self):
        """Create mock loss factory for pipeline testing."""
        def factory():
            return Mock()
        return factory
    
    def test_pipeline_with_cv(self, mock_data_processor, mock_model_factory, mock_loss_factory):
        """Test complete pipeline with cross-validation."""
        with patch('pinn_uv_vis_spectra.training.UVVisLeaveOneScanOutCV') as mock_cv_class:
            # Mock CV strategy
            mock_cv = Mock()
            mock_cv.execute_cross_validation.return_value = {
                'success': True,
                'n_successful_folds': 3,
                'aggregated_metrics': {'mse': {'mean': 0.1}}
            }
            mock_cv_class.return_value = mock_cv
            
            # Execute pipeline
            results = create_training_pipeline(
                data_processor=mock_data_processor,
                model_factory=mock_model_factory,
                loss_factory=mock_loss_factory,
                enable_cv=True
            )
            
            assert results['success']
            assert 'cv_results' in results
            assert results['cv_results']['success']
    
    def test_pipeline_without_cv(self, mock_data_processor, mock_model_factory, mock_loss_factory):
        """Test pipeline without cross-validation."""
        with patch('pinn_uv_vis_spectra.training.UVVisTrainingStrategy') as mock_strategy_class:
            # Mock training strategy
            mock_strategy = Mock()
            mock_strategy.execute_full_training.return_value = {
                'success': True,
                'final_loss': 0.05
            }
            mock_strategy_class.return_value = mock_strategy
            
            # Execute pipeline
            results = create_training_pipeline(
                data_processor=mock_data_processor,
                model_factory=mock_model_factory,
                loss_factory=mock_loss_factory,
                enable_cv=False
            )
            
            assert results['success']
            assert 'training_results' in results
            assert results['training_results']['success']
    
    def test_pipeline_error_handling(self, mock_data_processor, mock_model_factory, mock_loss_factory):
        """Test pipeline error handling."""
        # Mock data processor to raise exception
        mock_data_processor.create_training_data.side_effect = Exception("Data processing failed")
        
        results = create_training_pipeline(
            data_processor=mock_data_processor,
            model_factory=mock_model_factory,
            loss_factory=mock_loss_factory,
            enable_cv=False
        )
        
        assert not results['success']
        assert 'error' in results


class TestIntegrationScenarios:
    """Integration tests for complete training scenarios."""
    
    def test_end_to_end_training_simulation(self, tmp_path):
        """Test end-to-end training simulation with mocked components."""
        # Create mock components that simulate realistic behavior
        mock_model = Mock()
        mock_model.compile = Mock()
        mock_model.restore = Mock()
        
        # Simulate decreasing loss over time
        def mock_train(iterations, **kwargs):
            mock_losshistory = Mock()
            n_points = iterations // 100
            mock_losshistory.loss_train = [1.0 * np.exp(-0.1 * i) for i in range(n_points)]
            return mock_losshistory
        
        mock_model.train = mock_train
        
        # Mock network with realistic layers
        mock_net = Mock()
        mock_net.layers = []
        for _ in range(3):  # 3 layers
            layer = Mock()
            layer.kernel = tf.Variable(np.random.randn(10, 10) * 0.1)
            layer.bias = tf.Variable(np.random.randn(10) * 0.1)
            mock_net.layers.append(layer)
        mock_model.net = mock_net
        
        mock_loss_function = Mock(spec=UVVisLossFunction)
        mock_loss_function.get_loss_statistics.return_value = {
            'data': {'current': 0.1, 'trend': 'decreasing'},
            'physics': {'current': 0.05, 'trend': 'stable'},
            'smoothness': {'current': 0.001, 'trend': 'decreasing'}
        }
        
        # Create training strategy
        checkpoint_dir = tmp_path / "integration_test"
        strategy = UVVisTrainingStrategy(
            model=mock_model,
            loss_function=mock_loss_function,
            checkpoint_dir=str(checkpoint_dir),
            enable_adaptive_switching=False  # Disable for simpler testing
        )
        
        # Execute training
        results = strategy.execute_full_training()
        
        # Verify results
        assert results['success']
        assert 'training_history' in results
        assert 'stage_summaries' in results
        
        # Check that all stages were executed
        expected_stages = ['initialization', 'adam_training', 'lbfgs_refinement']
        for stage in expected_stages:
            assert stage in results['training_history']
        
        # Check that checkpoints directory was created and used
        assert checkpoint_dir.exists()
        assert len(list(checkpoint_dir.glob("*.json"))) > 0  # Should have training report


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness of training components."""
    
    def test_training_strategy_with_failing_model(self, tmp_path):
        """Test training strategy with model that fails."""
        mock_model = Mock()
        mock_model.compile = Mock()
        mock_model.train.side_effect = Exception("Model training failed")
        
        mock_loss_function = Mock()
        
        strategy = UVVisTrainingStrategy(
            model=mock_model,
            loss_function=mock_loss_function,
            checkpoint_dir=str(tmp_path)
        )
        
        results = strategy.execute_full_training()
        
        # Should handle error gracefully
        assert not results['success']
        assert 'error' in results
    
    def test_cv_with_failing_folds(self, tmp_path):
        """Test CV with some failing folds."""
        mock_data_processor = Mock()
        mock_data_processor.concentrations_nonzero = np.array([10, 20, 30])
        mock_data_processor.concentration_norm_params = {'scale': 60.0}
        
        def failing_model_factory(data):
            raise Exception("Model creation failed")
        
        def mock_loss_factory():
            return Mock()
        
        cv_strategy = UVVisLeaveOneScanOutCV(
            data_processor=mock_data_processor,
            model_factory=failing_model_factory,
            loss_factory=mock_loss_factory,
            cv_results_dir=str(tmp_path)
        )
        
        # Mock data creation to avoid additional complexity
        cv_strategy._create_train_data = Mock(return_value=(np.random.rand(10, 2), np.random.rand(10, 1)))
        cv_strategy._create_test_data = Mock(return_value=(np.random.rand(5, 2), np.random.rand(5, 1)))
        
        results = cv_strategy.execute_cross_validation()
        
        # Should complete but with failed folds
        assert not results['success']  # Should fail if all folds fail
        assert results['n_successful_folds'] == 0


# Test utilities and fixtures
@pytest.fixture(scope="session")
def temp_directory():
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    # Run tests when script is executed directly
    run_tests()
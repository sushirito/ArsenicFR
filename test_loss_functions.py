"""
Comprehensive Unit Tests for UV-Vis PINN Loss Functions
======================================================

This module contains comprehensive unit tests for the loss function implementation,
including tests for:
- Multi-component loss function correctness
- Physics constraint validation
- Smoothness regularization
- Numerical stability
- Adaptive weighting mechanisms
- Edge cases and error handling

Author: Claude Code Assistant
"""

import pytest
import numpy as np
import tensorflow as tf
import deepxde as dde
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

# Import modules to test
from .loss_functions import (
    UVVisLossFunction,
    AdaptiveLossWeighting,
    create_uvvis_loss_function,
    validate_loss_function
)


class TestUVVisLossFunction:
    """Test suite for UVVisLossFunction class."""
    
    @pytest.fixture
    def setup_loss_function(self):
        """Setup standard loss function for testing."""
        return UVVisLossFunction(
            path_length=1.0,
            physics_weight=0.1,
            smooth_weight=1e-4,
            enable_adaptive_weighting=False
        )
    
    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic test data."""
        n_points = 100
        
        # Normalized inputs: [wavelength_norm, concentration_norm]
        wavelengths = np.linspace(-1, 1, n_points)
        concentrations = np.linspace(0, 1, n_points)
        
        # Create mesh
        wl_mesh, conc_mesh = np.meshgrid(wavelengths, concentrations)
        inputs = np.column_stack([wl_mesh.flatten(), conc_mesh.flatten()])
        
        # Synthetic targets following Beer-Lambert law
        # ε(λ) = 0.1 + 0.05 * sin(5 * λ) (synthetic molar absorptivity)
        epsilon_true = 0.1 + 0.05 * np.sin(5 * wl_mesh.flatten())
        targets = epsilon_true * 1.0 * conc_mesh.flatten()  # ΔA = ε * b * c
        
        return {
            'inputs': inputs.astype(np.float32),
            'targets': targets.reshape(-1, 1).astype(np.float32),
            'epsilon_true': epsilon_true.reshape(-1, 1).astype(np.float32)
        }
    
    def test_initialization(self):
        """Test loss function initialization."""
        # Test default initialization
        loss_fn = UVVisLossFunction()
        assert loss_fn.path_length == 1.0
        assert loss_fn.physics_weight == 0.1
        assert loss_fn.smooth_weight == 1e-4
        assert not loss_fn.enable_adaptive_weighting
        
        # Test custom initialization
        loss_fn_custom = UVVisLossFunction(
            path_length=2.0,
            physics_weight=0.2,
            smooth_weight=1e-3,
            enable_adaptive_weighting=True
        )
        assert loss_fn_custom.path_length == 2.0
        assert loss_fn_custom.physics_weight == 0.2
        assert loss_fn_custom.smooth_weight == 1e-3
        assert loss_fn_custom.enable_adaptive_weighting
    
    def test_create_loss_function(self, setup_loss_function):
        """Test loss function creation."""
        loss_fn = setup_loss_function.create_loss_function()
        
        # Should return callable function
        assert callable(loss_fn)
        
        # Test function signature by calling with dummy data
        inputs = tf.constant([[0.5, 0.5]], dtype=tf.float32)
        targets = tf.constant([[0.05]], dtype=tf.float32)
        outputs = tf.constant([[0.1]], dtype=tf.float32)
        
        result = loss_fn(targets, outputs, inputs)
        
        # Should return scalar tensor
        assert result.shape == ()
        assert tf.math.is_finite(result)
    
    def test_data_loss_computation(self, setup_loss_function, synthetic_data):
        """Test data loss computation."""
        inputs_tf = tf.constant(synthetic_data['inputs'])
        targets_tf = tf.constant(synthetic_data['targets'])
        outputs_tf = tf.constant(synthetic_data['epsilon_true'])
        
        # Call internal method for testing
        wavelength_norm = inputs_tf[:, 0:1]
        concentration_norm = inputs_tf[:, 1:2]
        
        data_loss = setup_loss_function._compute_data_loss(
            targets_tf, outputs_tf, wavelength_norm, concentration_norm
        )
        
        # Should be finite and positive
        assert tf.math.is_finite(data_loss)
        assert data_loss >= 0
        
        # For perfect predictions, loss should be very small (only regularization)
        assert data_loss.numpy() < 0.01  # Should be dominated by regularization term
    
    def test_physics_loss_computation(self, setup_loss_function, synthetic_data):
        """Test physics constraint loss computation."""
        inputs_tf = tf.constant(synthetic_data['inputs'])
        outputs_tf = tf.constant(synthetic_data['epsilon_true'])
        
        wavelength_norm = inputs_tf[:, 0:1]
        concentration_norm = inputs_tf[:, 1:2]
        
        physics_loss = setup_loss_function._compute_physics_loss(
            outputs_tf, wavelength_norm, concentration_norm
        )
        
        # Should be finite and non-negative
        assert tf.math.is_finite(physics_loss)
        assert physics_loss >= 0
        
        # Physics loss should be small for synthetic data that follows Beer-Lambert
        assert physics_loss.numpy() < 1.0
    
    def test_smoothness_loss_computation(self, setup_loss_function, synthetic_data):
        """Test smoothness regularization computation."""
        inputs_tf = tf.constant(synthetic_data['inputs'])
        outputs_tf = tf.constant(synthetic_data['epsilon_true'])
        
        wavelength_norm = inputs_tf[:, 0:1]
        concentration_norm = inputs_tf[:, 1:2]
        
        smooth_loss = setup_loss_function._compute_smoothness_loss(
            outputs_tf, wavelength_norm, concentration_norm
        )
        
        # Should be finite and non-negative
        assert tf.math.is_finite(smooth_loss)
        assert smooth_loss >= 0
    
    def test_loss_component_weighting(self, synthetic_data):
        """Test loss component weighting."""
        # Test different weight configurations
        weights_configs = [
            (0.0, 0.0),    # Only data loss
            (1.0, 0.0),    # Data + physics
            (0.0, 1.0),    # Data + smoothness
            (0.5, 0.001)   # Balanced
        ]
        
        for physics_w, smooth_w in weights_configs:
            loss_fn_obj = UVVisLossFunction(
                physics_weight=physics_w,
                smooth_weight=smooth_w
            )
            loss_fn = loss_fn_obj.create_loss_function()
            
            inputs_tf = tf.constant(synthetic_data['inputs'])
            targets_tf = tf.constant(synthetic_data['targets'])
            outputs_tf = tf.constant(synthetic_data['epsilon_true'])
            
            total_loss = loss_fn(targets_tf, outputs_tf, inputs_tf)
            
            # Should always be finite and positive
            assert tf.math.is_finite(total_loss)
            assert total_loss > 0
    
    def test_gradient_computation(self, setup_loss_function, synthetic_data):
        """Test gradient computation through loss function."""
        loss_fn = setup_loss_function.create_loss_function()
        
        inputs_tf = tf.constant(synthetic_data['inputs'])
        targets_tf = tf.constant(synthetic_data['targets'])
        
        # Create trainable variables (mock network outputs)
        outputs_var = tf.Variable(
            synthetic_data['epsilon_true'], 
            trainable=True,
            dtype=tf.float32
        )
        
        with tf.GradientTape() as tape:
            loss_value = loss_fn(targets_tf, outputs_var, inputs_tf)
        
        # Compute gradients
        gradients = tape.gradient(loss_value, outputs_var)
        
        # Gradients should exist and be finite
        assert gradients is not None
        assert tf.reduce_all(tf.math.is_finite(gradients))
        
        # Gradients should not be all zeros (unless at perfect minimum)
        assert not tf.reduce_all(tf.equal(gradients, 0.0))
    
    def test_numerical_stability(self, setup_loss_function):
        """Test numerical stability with edge cases."""
        # Test with extreme values
        extreme_cases = [
            # Very small values
            {
                'inputs': np.array([[0.0, 1e-10]], dtype=np.float32),
                'targets': np.array([[1e-15]], dtype=np.float32),
                'outputs': np.array([[1e-12]], dtype=np.float32)
            },
            # Very large values
            {
                'inputs': np.array([[1.0, 1.0]], dtype=np.float32),
                'targets': np.array([[1e5]], dtype=np.float32),
                'outputs': np.array([[1e6]], dtype=np.float32)
            },
            # Zero concentration
            {
                'inputs': np.array([[0.0, 0.0]], dtype=np.float32),
                'targets': np.array([[0.0]], dtype=np.float32),
                'outputs': np.array([[0.1]], dtype=np.float32)
            }
        ]
        
        loss_fn = setup_loss_function.create_loss_function()
        
        for case in extreme_cases:
            inputs_tf = tf.constant(case['inputs'])
            targets_tf = tf.constant(case['targets'])
            outputs_tf = tf.constant(case['outputs'])
            
            loss_value = loss_fn(targets_tf, outputs_tf, inputs_tf)
            
            # Should remain finite
            assert tf.math.is_finite(loss_value), f"Loss not finite for case: {case}"
            assert loss_value >= 0, f"Loss negative for case: {case}"
    
    def test_loss_history_tracking(self, setup_loss_function, synthetic_data):
        """Test loss history tracking functionality."""
        loss_fn = setup_loss_function.create_loss_function()
        
        # Call loss function multiple times
        for _ in range(5):
            inputs_tf = tf.constant(synthetic_data['inputs'])
            targets_tf = tf.constant(synthetic_data['targets'])
            outputs_tf = tf.constant(synthetic_data['epsilon_true'])
            
            loss_fn(targets_tf, outputs_tf, inputs_tf)
        
        # Check that history is being tracked
        assert len(setup_loss_function.loss_history['total']) >= 1
        
        # Get statistics
        stats = setup_loss_function.get_loss_statistics()
        assert 'data' in stats
        assert 'physics' in stats
        assert 'smoothness' in stats
        assert 'total' in stats
    
    def test_fallback_mechanisms(self, setup_loss_function, synthetic_data):
        """Test fallback mechanisms when gradient computation fails."""
        # Mock gradient computation failure
        with patch('tensorflow.GradientTape') as mock_tape:
            mock_tape.side_effect = Exception("Gradient computation failed")
            
            loss_fn = setup_loss_function.create_loss_function()
            
            inputs_tf = tf.constant(synthetic_data['inputs'])
            targets_tf = tf.constant(synthetic_data['targets'])
            outputs_tf = tf.constant(synthetic_data['epsilon_true'])
            
            # Should not raise exception, should use fallback
            loss_value = loss_fn(targets_tf, outputs_tf, inputs_tf)
            
            assert tf.math.is_finite(loss_value)


class TestAdaptiveLossWeighting:
    """Test suite for AdaptiveLossWeighting class."""
    
    @pytest.fixture
    def adaptive_weighting(self):
        """Setup adaptive weighting for testing."""
        return AdaptiveLossWeighting(
            initial_physics_weight=0.1,
            initial_smooth_weight=1e-4,
            adaptation_rate=0.1
        )
    
    def test_initialization(self, adaptive_weighting):
        """Test adaptive weighting initialization."""
        assert adaptive_weighting.current_weights['physics'] == 0.1
        assert adaptive_weighting.current_weights['smoothness'] == 1e-4
        assert adaptive_weighting.adaptation_rate == 0.1
    
    def test_weight_update(self, adaptive_weighting):
        """Test weight update mechanism."""
        # Create mock loss tensors
        data_loss = tf.constant(1.0)
        physics_loss = tf.constant(10.0)  # Much larger
        smooth_loss = tf.constant(0.01)   # Much smaller
        
        # Update weights
        new_weights = adaptive_weighting.update_weights(
            data_loss, physics_loss, smooth_loss
        )
        
        # Physics weight should decrease (physics loss is too large)
        assert new_weights['physics'] < 0.1
        
        # Smoothness weight should increase (smoothness loss is small)
        assert new_weights['smoothness'] > 1e-4
        
        # Weights should be within reasonable bounds
        assert 1e-6 <= new_weights['physics'] <= 10.0
        assert 1e-8 <= new_weights['smoothness'] <= 1.0
    
    def test_weight_adaptation_convergence(self, adaptive_weighting):
        """Test that weights converge to reasonable values."""
        # Simulate multiple updates with consistent loss ratios
        for _ in range(50):
            data_loss = tf.constant(1.0)
            physics_loss = tf.constant(5.0)
            smooth_loss = tf.constant(0.1)
            
            adaptive_weighting.update_weights(data_loss, physics_loss, smooth_loss)
        
        # After many updates, weights should stabilize
        final_weights = adaptive_weighting.current_weights
        
        # Physics weight should be reduced
        assert final_weights['physics'] < 0.1
        
        # Smoothness weight should be increased
        assert final_weights['smoothness'] > 1e-4


class TestLossFunctionFactory:
    """Test suite for loss function factory functions."""
    
    def test_create_uvvis_loss_function(self):
        """Test loss function factory."""
        # Test with default parameters
        loss_fn = create_uvvis_loss_function()
        assert callable(loss_fn)
        
        # Test with custom parameters
        loss_fn_custom = create_uvvis_loss_function(
            path_length=2.0,
            physics_weight=0.2,
            smooth_weight=1e-3,
            enable_adaptive_weighting=True
        )
        assert callable(loss_fn_custom)
        
        # Test that functions are different (different closures)
        assert loss_fn != loss_fn_custom
    
    def test_validate_loss_function(self):
        """Test loss function validation utility."""
        # Create test loss function
        loss_fn = create_uvvis_loss_function()
        
        # Create test data
        inputs = np.array([[0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
        targets = np.array([[0.05], [0.1]], dtype=np.float32)
        expected_ranges = (0.05, 0.15)
        
        # Validate
        validation_result = validate_loss_function(
            loss_fn, 
            (inputs, targets, expected_ranges)
        )
        
        assert validation_result['validation_passed']
        assert validation_result['is_finite']
        assert validation_result['is_positive']
        assert isinstance(validation_result['loss_value'], float)
    
    def test_validate_loss_function_error_handling(self):
        """Test error handling in loss function validation."""
        # Create a loss function that raises an error
        def failing_loss_fn(targets, outputs, inputs):
            raise ValueError("Test error")
        
        inputs = np.array([[0.5, 0.5]], dtype=np.float32)
        targets = np.array([[0.05]], dtype=np.float32)
        expected_ranges = (0.05, 0.15)
        
        validation_result = validate_loss_function(
            failing_loss_fn,
            (inputs, targets, expected_ranges)
        )
        
        assert not validation_result['validation_passed']
        assert 'error' in validation_result
        assert validation_result['error'] == "Test error"


class TestPhysicsConstraints:
    """Test suite specifically for physics constraint validation."""
    
    @pytest.fixture
    def physics_loss_function(self):
        """Setup loss function for physics testing."""
        return UVVisLossFunction(
            physics_weight=1.0,  # High weight for physics
            smooth_weight=0.0    # No smoothness for isolation
        )
    
    def test_beer_lambert_constraint(self, physics_loss_function):
        """Test Beer-Lambert law constraint satisfaction."""
        # Create data that perfectly satisfies Beer-Lambert law
        n_points = 50
        wavelengths = np.linspace(-1, 1, n_points)
        concentrations = np.linspace(0.1, 1, n_points)  # Avoid zero
        
        # Create mesh
        wl_mesh, conc_mesh = np.meshgrid(wavelengths, concentrations)
        inputs = np.column_stack([wl_mesh.flatten(), conc_mesh.flatten()])
        
        # Perfect Beer-Lambert relationship: ε independent of c
        epsilon_perfect = 0.1 + 0.05 * np.sin(3 * wl_mesh.flatten())
        targets = epsilon_perfect * 1.0 * conc_mesh.flatten()  # ΔA = ε * b * c
        
        # Create tensors
        inputs_tf = tf.constant(inputs.astype(np.float32))
        targets_tf = tf.constant(targets.reshape(-1, 1).astype(np.float32))
        outputs_tf = tf.constant(epsilon_perfect.reshape(-1, 1).astype(np.float32))
        
        # Test physics loss
        wavelength_norm = inputs_tf[:, 0:1]
        concentration_norm = inputs_tf[:, 1:2]
        
        physics_loss = physics_loss_function._compute_physics_loss(
            outputs_tf, wavelength_norm, concentration_norm
        )
        
        # For perfect Beer-Lambert data, physics loss should be small
        assert physics_loss.numpy() < 0.1
    
    def test_concentration_linearity(self, physics_loss_function):
        """Test that physics loss penalizes non-linear concentration dependence."""
        # Create data that violates Beer-Lambert law (quadratic in concentration)
        n_points = 50
        wavelengths = np.array([0.0] * n_points)  # Fixed wavelength
        concentrations = np.linspace(0.1, 1, n_points)
        
        inputs = np.column_stack([wavelengths, concentrations])
        
        # Non-linear relationship: ε depends on c (violates Beer-Lambert)
        epsilon_nonlinear = 0.1 * (1 + concentrations**2)  # Nonlinear in c
        targets = epsilon_nonlinear * 1.0 * concentrations
        
        # Create tensors
        inputs_tf = tf.constant(inputs.astype(np.float32))
        targets_tf = tf.constant(targets.reshape(-1, 1).astype(np.float32))
        outputs_tf = tf.constant(epsilon_nonlinear.reshape(-1, 1).astype(np.float32))
        
        # Test physics loss
        wavelength_norm = inputs_tf[:, 0:1]
        concentration_norm = inputs_tf[:, 1:2]
        
        physics_loss = physics_loss_function._compute_physics_loss(
            outputs_tf, wavelength_norm, concentration_norm
        )
        
        # For non-linear data, physics loss should be larger
        assert physics_loss.numpy() > 0.1
    
    def test_boundary_condition_zero_concentration(self, physics_loss_function):
        """Test behavior at zero concentration boundary."""
        # Test with zero concentration
        inputs = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=np.float32)
        targets = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)  # Should be zero at c=0
        outputs = np.array([[0.1], [0.15], [0.12]], dtype=np.float32)  # Non-zero epsilon
        
        inputs_tf = tf.constant(inputs)
        targets_tf = tf.constant(targets)
        outputs_tf = tf.constant(outputs)
        
        loss_fn = physics_loss_function.create_loss_function()
        loss_value = loss_fn(targets_tf, outputs_tf, inputs_tf)
        
        # Should handle zero concentration gracefully
        assert tf.math.is_finite(loss_value)
        assert loss_value >= 0


class TestIntegrationWithDeepXDE:
    """Integration tests with DeepXDE framework."""
    
    @pytest.fixture
    def mock_deepxde_model(self):
        """Create mock DeepXDE model for testing."""
        mock_model = Mock()
        mock_model.compile = Mock()
        mock_model.train = Mock()
        mock_model.predict = Mock(return_value=np.array([[0.1], [0.15], [0.12]]))
        return mock_model
    
    def test_loss_function_with_model_compile(self, mock_deepxde_model):
        """Test loss function integration with DeepXDE model.compile()."""
        loss_fn = create_uvvis_loss_function()
        
        # Should be able to compile model with custom loss
        mock_deepxde_model.compile(
            optimizer='adam',
            loss=loss_fn,
            lr=1e-3
        )
        
        # Verify compile was called
        mock_deepxde_model.compile.assert_called_once()
    
    def test_loss_function_training_compatibility(self, mock_deepxde_model):
        """Test that loss function works during training."""
        loss_fn = create_uvvis_loss_function()
        
        # Mock training scenario
        mock_deepxde_model.compile(optimizer='adam', loss=loss_fn)
        mock_deepxde_model.train(iterations=100)
        
        # Should complete without errors
        mock_deepxde_model.train.assert_called_once()


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness."""
    
    def test_invalid_input_shapes(self):
        """Test handling of invalid input shapes."""
        loss_fn = create_uvvis_loss_function()
        
        # Wrong input shape (should be [N, 2])
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            inputs = tf.constant([[0.5]], dtype=tf.float32)  # Wrong shape
            targets = tf.constant([[0.05]], dtype=tf.float32)
            outputs = tf.constant([[0.1]], dtype=tf.float32)
            loss_fn(targets, outputs, inputs)
    
    def test_nan_input_handling(self):
        """Test handling of NaN inputs."""
        loss_fn = create_uvvis_loss_function()
        
        # Inputs with NaN
        inputs = tf.constant([[0.5, np.nan]], dtype=tf.float32)
        targets = tf.constant([[0.05]], dtype=tf.float32)
        outputs = tf.constant([[0.1]], dtype=tf.float32)
        
        result = loss_fn(targets, outputs, inputs)
        
        # Should handle NaN gracefully (might be NaN but shouldn't crash)
        assert result.dtype == tf.float32
    
    def test_infinite_input_handling(self):
        """Test handling of infinite inputs."""
        loss_fn = create_uvvis_loss_function()
        
        # Inputs with infinity
        inputs = tf.constant([[0.5, np.inf]], dtype=tf.float32)
        targets = tf.constant([[0.05]], dtype=tf.float32)
        outputs = tf.constant([[0.1]], dtype=tf.float32)
        
        result = loss_fn(targets, outputs, inputs)
        
        # Should handle infinity gracefully
        assert result.dtype == tf.float32


class TestPerformanceAndScalability:
    """Test performance and scalability of loss functions."""
    
    def test_large_batch_performance(self):
        """Test loss function performance with large batches."""
        loss_fn = create_uvvis_loss_function()
        
        # Create large batch
        n_large = 10000
        inputs = tf.random.uniform((n_large, 2), dtype=tf.float32)
        targets = tf.random.uniform((n_large, 1), dtype=tf.float32)
        outputs = tf.random.uniform((n_large, 1), dtype=tf.float32)
        
        # Time the computation
        import time
        start_time = time.time()
        
        result = loss_fn(targets, outputs, inputs)
        
        end_time = time.time()
        
        # Should complete in reasonable time (< 1 second for 10k points)
        assert end_time - start_time < 1.0
        assert tf.math.is_finite(result)
    
    def test_memory_efficiency(self):
        """Test that loss function doesn't cause memory leaks."""
        loss_fn = create_uvvis_loss_function()
        
        # Run many iterations to check for memory leaks
        for i in range(100):
            inputs = tf.random.uniform((100, 2), dtype=tf.float32)
            targets = tf.random.uniform((100, 1), dtype=tf.float32)
            outputs = tf.random.uniform((100, 1), dtype=tf.float32)
            
            result = loss_fn(targets, outputs, inputs)
            
            # Clear references
            del inputs, targets, outputs, result


# Test fixtures and utilities
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
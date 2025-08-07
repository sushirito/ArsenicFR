"""
Unit Tests for Phase 2: Model Definition
========================================

Comprehensive test suite for PINN model architecture, DeepXDE integration,
and physics constraint implementation.

Author: Claude Code Assistant
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Mock deepxde to avoid import errors during testing
sys.modules['deepxde'] = MagicMock()
sys.modules['deepxde.nn'] = MagicMock()
sys.modules['deepxde.geometry'] = MagicMock()
sys.modules['deepxde.data'] = MagicMock()
sys.modules['deepxde.icbc'] = MagicMock()
sys.modules['deepxde.model'] = MagicMock()
sys.modules['deepxde.grad'] = MagicMock()
sys.modules['deepxde.backend'] = MagicMock()

from model_definition import SpectroscopyPINN, create_spectroscopy_pinn


class TestSpectroscopyPINN:
    """Test suite for SpectroscopyPINN class."""
    
    @pytest.fixture
    def basic_pinn(self):
        """Create basic PINN instance for testing."""
        return SpectroscopyPINN(
            wavelength_range=(200, 800),
            concentration_range=(0, 60),
            layer_sizes=[2, 32, 64, 32, 1],
            activation="tanh"
        )
    
    @pytest.fixture
    def custom_pinn(self):
        """Create custom PINN with different parameters."""
        return SpectroscopyPINN(
            wavelength_range=(250, 750),
            concentration_range=(0, 100),
            layer_sizes=[2, 64, 128, 64, 1],
            activation="swish",
            kernel_initializer="He normal"
        )
    
    def test_initialization_basic(self, basic_pinn):
        """Test basic PINN initialization."""
        assert basic_pinn.wavelength_range == (200, 800)
        assert basic_pinn.concentration_range == (0, 60)
        assert basic_pinn.layer_sizes == [2, 32, 64, 32, 1]
        assert basic_pinn.activation == "tanh"
        assert basic_pinn.kernel_initializer == "Glorot normal"
        assert basic_pinn.path_length == 1.0
        
        # Initially None components
        assert basic_pinn.geometry is None
        assert basic_pinn.net is None
        assert basic_pinn.pde is None
        assert basic_pinn.data is None
        assert basic_pinn.model is None
    
    def test_initialization_custom(self, custom_pinn):
        """Test custom PINN initialization."""
        assert custom_pinn.wavelength_range == (250, 750)
        assert custom_pinn.concentration_range == (0, 100)
        assert custom_pinn.layer_sizes == [2, 64, 128, 64, 1]
        assert custom_pinn.activation == "swish"
        assert custom_pinn.kernel_initializer == "He normal"
    
    @patch('model_definition.dde.geometry.Rectangle')
    def test_create_geometry(self, mock_rectangle, basic_pinn):
        """Test geometry creation."""
        mock_geom = MagicMock()
        mock_rectangle.return_value = mock_geom
        
        geometry = basic_pinn.create_geometry()
        
        # Check that Rectangle was called with correct normalized bounds
        mock_rectangle.assert_called_once_with(
            xmin=[-1.0, 0.0],
            xmax=[1.0, 1.0]
        )
        
        assert basic_pinn.geometry is mock_geom
        assert geometry is mock_geom
    
    def test_beer_lambert_pde_structure(self, basic_pinn):
        """Test Beer-Lambert PDE function structure and inputs."""
        # Create mock input arrays
        n_points = 100
        x_mock = np.random.rand(n_points, 2)  # [lambda_norm, c_norm]
        y_mock = np.random.rand(n_points, 1)  # delta_A
        
        # Mock gradient functions
        with patch('model_definition.dde.grad.jacobian') as mock_jacobian:
            mock_jacobian.return_value = np.random.rand(n_points, 1)
            
            residual = basic_pinn.beer_lambert_pde(x_mock, y_mock)
            
            # Should call jacobian for computing derivatives
            assert mock_jacobian.called
            assert residual is not None
            
            # Check that jacobian was called with correct parameters for concentration derivative
            jacobian_calls = mock_jacobian.call_args_list
            assert len(jacobian_calls) >= 1  # At least one gradient computation
    
    @patch('model_definition.dde.nn.FNN')
    def test_create_neural_network(self, mock_fnn, basic_pinn):
        """Test neural network creation and configuration."""
        mock_net = MagicMock()
        mock_fnn.return_value = mock_net
        
        network = basic_pinn.create_neural_network()
        
        # Check FNN was called with correct parameters
        mock_fnn.assert_called_once_with(
            layer_sizes=[2, 32, 64, 32, 1],
            activation="tanh",
            kernel_initializer="Glorot normal"
        )
        
        # Check transformations were applied
        mock_net.apply_input_transform.assert_called_once()
        mock_net.apply_output_transform.assert_called_once()
        
        assert basic_pinn.net is mock_net
        assert network is mock_net
    
    def test_neural_network_input_transform(self, basic_pinn):
        """Test input transformation function."""
        # Get the input transform function by creating network
        with patch('model_definition.dde.nn.FNN') as mock_fnn:
            mock_net = MagicMock()
            mock_fnn.return_value = mock_net
            
            basic_pinn.create_neural_network()
            
            # Get the input transform function from the call
            input_transform = mock_net.apply_input_transform.call_args[0][0]
            
            # Test the transformation
            test_input = np.array([[0.5, 0.3], [-0.2, 0.8]])  # [lambda_norm, c_norm]
            
            with patch('model_definition.dde.backend.concat') as mock_concat:
                mock_concat.return_value = "transformed"
                result = input_transform(test_input)
                
                # Should call concat to combine transformed inputs
                mock_concat.assert_called_once()
                assert result == "transformed"
    
    def test_neural_network_output_transform(self, basic_pinn):
        """Test output transformation for physical constraints."""
        with patch('model_definition.dde.nn.FNN') as mock_fnn:
            mock_net = MagicMock()
            mock_fnn.return_value = mock_net
            
            basic_pinn.create_neural_network()
            
            # Get the output transform function
            output_transform = mock_net.apply_output_transform.call_args[0][0]
            
            # Test the transformation
            x_test = np.array([[0.5, 0.0], [0.3, 0.5]])  # [lambda_norm, c_norm]
            y_test = np.array([[1.0], [0.8]])  # raw network output
            
            result = output_transform(x_test, y_test)
            
            # Should enforce zero output when concentration is zero
            assert result is not None
            assert result.shape == y_test.shape
    
    @patch('model_definition.dde.icbc.DirichletBC')
    def test_create_boundary_conditions(self, mock_bc, basic_pinn):
        """Test boundary condition creation."""
        mock_bc_instance = MagicMock()
        mock_bc.return_value = mock_bc_instance
        
        basic_pinn.geometry = MagicMock()  # Set mock geometry
        boundary_conditions = basic_pinn._create_boundary_conditions()
        
        assert isinstance(boundary_conditions, list)
        assert len(boundary_conditions) == 1  # Zero concentration BC
        assert mock_bc.called
        assert boundary_conditions[0] is mock_bc_instance
    
    @patch('model_definition.dde.data.PDE')
    def test_create_pde_data(self, mock_pde, basic_pinn):
        """Test PDE data creation."""
        mock_pde_instance = MagicMock()
        mock_pde.return_value = mock_pde_instance
        
        # Create geometry first
        basic_pinn.geometry = MagicMock()
        
        pde_data = basic_pinn.create_pde_data(
            num_domain=1000,
            num_boundary=100,
            num_test=200
        )
        
        # Check PDE was called with correct parameters
        mock_pde.assert_called_once()
        call_args = mock_pde.call_args
        
        assert call_args[1]['geometry'] is basic_pinn.geometry
        assert call_args[1]['pde'] == basic_pinn.beer_lambert_pde
        assert call_args[1]['num_domain'] == 1000
        assert call_args[1]['num_boundary'] == 100
        assert call_args[1]['num_test'] == 200
        assert call_args[1]['train_distribution'] == 'Hammersley'
        
        assert basic_pinn.pde is mock_pde_instance
        assert pde_data is mock_pde_instance
    
    @patch('model_definition.dde.data.PDE')
    @patch('model_definition.dde.data.DataSet')
    @patch('model_definition.dde.data.combine.CombinedData')
    def test_create_combined_data_with_experimental(self, mock_combined, mock_dataset, mock_pde, basic_pinn):
        """Test combined data creation with experimental data."""
        # Setup mocks
        mock_pde_instance = MagicMock()
        mock_pde.return_value = mock_pde_instance
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance
        mock_combined_instance = MagicMock()
        mock_combined.return_value = mock_combined_instance
        
        basic_pinn.geometry = MagicMock()
        
        # Create experimental data
        n_points = 100
        X_exp = np.random.rand(n_points, 2)
        y_exp = np.random.rand(n_points, 1)
        
        combined_data = basic_pinn.create_combined_data(
            experimental_data=(X_exp, y_exp)
        )
        
        # Check that DataSet was created
        mock_dataset.assert_called_once()
        
        # Check that CombinedData was created with both datasets
        mock_combined.assert_called_once()
        combined_args = mock_combined.call_args[0][0]
        assert len(combined_args) == 2  # supervised + PDE data
        
        assert basic_pinn.data is mock_combined_instance
        assert combined_data is mock_combined_instance
    
    @patch('model_definition.dde.data.PDE')
    def test_create_combined_data_physics_only(self, mock_pde, basic_pinn):
        """Test combined data creation without experimental data."""
        mock_pde_instance = MagicMock()
        mock_pde.return_value = mock_pde_instance
        
        basic_pinn.geometry = MagicMock()
        
        combined_data = basic_pinn.create_combined_data(experimental_data=None)
        
        # Should return PDE data directly
        assert basic_pinn.data is mock_pde_instance
        assert combined_data is mock_pde_instance
    
    @patch('model_definition.dde.Model')
    def test_create_model(self, mock_model, basic_pinn):
        """Test complete model creation."""
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Setup prerequisites
        with patch.object(basic_pinn, 'create_neural_network') as mock_create_net, \
             patch.object(basic_pinn, 'create_combined_data') as mock_create_data:
            
            mock_net = MagicMock()
            mock_data = MagicMock()
            mock_create_net.return_value = mock_net
            mock_create_data.return_value = mock_data
            
            # Ensure the network is set on the instance
            def set_net():
                basic_pinn.net = mock_net
                return mock_net
            mock_create_net.side_effect = set_net
            
            # Create experimental data
            X_exp = np.random.rand(50, 2)
            y_exp = np.random.rand(50, 1)
            
            model = basic_pinn.create_model(experimental_data=(X_exp, y_exp))
            
            # Check methods were called
            mock_create_net.assert_called_once()
            mock_create_data.assert_called_once_with((X_exp, y_exp))
            
            # Check Model was created
            mock_model.assert_called_once_with(mock_data, mock_net)
            
            assert basic_pinn.model is mock_model_instance
            assert model is mock_model_instance
    
    def test_parameter_counting(self):
        """Test parameter counting for different architectures."""
        # Test small network
        pinn_small = SpectroscopyPINN(layer_sizes=[2, 10, 1])
        expected_small = (2 * 10 + 10) + (10 * 1 + 1)  # weights + biases
        assert pinn_small._count_parameters() == expected_small
        
        # Test larger network
        pinn_large = SpectroscopyPINN(layer_sizes=[2, 64, 128, 64, 1])
        expected_large = ((2*64 + 64) + (64*128 + 128) + 
                         (128*64 + 64) + (64*1 + 1))
        assert pinn_large._count_parameters() == expected_large
        
        # Test empty network
        pinn_empty = SpectroscopyPINN(layer_sizes=[])
        assert pinn_empty._count_parameters() == 0
    
    def test_get_model_summary(self, basic_pinn):
        """Test model summary generation."""
        summary = basic_pinn.get_model_summary()
        
        # Check required sections
        assert "architecture" in summary
        assert "domain" in summary
        assert "physics" in summary
        assert "geometry" in summary
        
        # Check architecture details
        arch = summary["architecture"]
        assert arch["layer_sizes"] == [2, 32, 64, 32, 1]
        assert arch["activation"] == "tanh"
        assert arch["kernel_initializer"] == "Glorot normal"
        assert "total_parameters" in arch
        
        # Check domain details
        domain = summary["domain"]
        assert domain["wavelength_range"] == (200, 800)
        assert domain["concentration_range"] == (0, 60)
        assert domain["path_length"] == 1.0
        
        # Check physics details
        physics = summary["physics"]
        assert "pde_constraint" in physics
        assert "boundary_conditions" in physics
        assert "regularization" in physics


class TestFactoryFunction:
    """Test suite for factory function."""
    
    def test_create_spectroscopy_pinn_standard(self):
        """Test standard architecture creation."""
        pinn = create_spectroscopy_pinn(architecture="standard")
        
        assert isinstance(pinn, SpectroscopyPINN)
        assert pinn.layer_sizes == [2, 64, 128, 128, 64, 32, 1]
        assert pinn.activation == "tanh"
        assert pinn.wavelength_range == (200, 800)
        assert pinn.concentration_range == (0, 60)
    
    def test_create_spectroscopy_pinn_deep(self):
        """Test deep architecture creation."""
        pinn = create_spectroscopy_pinn(architecture="deep")
        
        assert pinn.layer_sizes == [2, 64, 128, 256, 256, 128, 64, 32, 1]
        assert pinn.activation == "tanh"
    
    def test_create_spectroscopy_pinn_wide(self):
        """Test wide architecture creation."""
        pinn = create_spectroscopy_pinn(architecture="wide")
        
        assert pinn.layer_sizes == [2, 128, 256, 256, 128, 1]
        assert pinn.activation == "swish"
    
    def test_create_spectroscopy_pinn_custom_ranges(self):
        """Test creation with custom ranges."""
        pinn = create_spectroscopy_pinn(
            wavelength_range=(300, 700),
            concentration_range=(0, 100),
            architecture="standard"
        )
        
        assert pinn.wavelength_range == (300, 700)
        assert pinn.concentration_range == (0, 100)
    
    def test_create_spectroscopy_pinn_invalid_architecture(self):
        """Test error handling for invalid architecture."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_spectroscopy_pinn(architecture="invalid")


class TestPhysicsConstraints:
    """Test physics-related aspects of the model."""
    
    @pytest.fixture
    def physics_pinn(self):
        """Create PINN for physics testing."""
        return SpectroscopyPINN(
            wavelength_range=(200, 800),
            concentration_range=(0, 60),
            layer_sizes=[2, 16, 32, 16, 1],
            activation="tanh"
        )
    
    def test_output_transform_zero_concentration(self, physics_pinn):
        """Test that output transform enforces zero absorption at zero concentration."""
        with patch('model_definition.dde.nn.FNN') as mock_fnn:
            mock_net = MagicMock()
            mock_fnn.return_value = mock_net
            
            physics_pinn.create_neural_network()
            
            # Verify the method was called and get the output transform function
            mock_net.apply_output_transform.assert_called_once()
            output_transform = mock_net.apply_output_transform.call_args[0][0]
            
            # Test with zero concentration
            x_zero_conc = np.array([[0.0, 0.0], [0.5, 0.0]])  # c_norm = 0
            y_raw = np.array([[1.0], [0.8]])  # Non-zero raw output
            
            result = output_transform(x_zero_conc, y_raw)
            
            # Output should be zero when concentration is zero
            expected = np.array([[0.0], [0.0]])
            np.testing.assert_array_equal(result, expected)
    
    def test_output_transform_nonzero_concentration(self, physics_pinn):
        """Test output transform with non-zero concentrations."""
        with patch('model_definition.dde.nn.FNN') as mock_fnn:
            mock_net = MagicMock()
            mock_fnn.return_value = mock_net
            
            physics_pinn.create_neural_network()
            
            # Verify the method was called and get the output transform function
            mock_net.apply_output_transform.assert_called_once()
            output_transform = mock_net.apply_output_transform.call_args[0][0]
            
            # Test with non-zero concentrations
            x_nonzero = np.array([[0.0, 0.5], [0.5, 1.0]])  # c_norm > 0
            y_raw = np.array([[1.0], [0.8]])
            
            result = output_transform(x_nonzero, y_raw)
            
            # Output should be scaled by concentration
            expected = np.array([[0.5], [0.8]])
            np.testing.assert_array_equal(result, expected)
    
    def test_boundary_condition_functions(self, physics_pinn):
        """Test boundary condition callback functions."""
        physics_pinn.geometry = MagicMock()
        
        boundary_conditions = physics_pinn._create_boundary_conditions()
        
        # Should have one boundary condition
        assert len(boundary_conditions) == 1
    
    def test_path_length_parameter(self, physics_pinn):
        """Test path length parameter in Beer-Lambert calculations."""
        assert physics_pinn.path_length == 1.0  # Default cuvette path length
        
        # Test with custom path length
        custom_pinn = SpectroscopyPINN(layer_sizes=[2, 10, 1])
        custom_pinn.path_length = 0.5  # Half-cm cuvette
        assert custom_pinn.path_length == 0.5


class TestIntegrationScenarios:
    """Test integrated scenarios and workflows."""
    
    def test_complete_workflow_simulation(self):
        """Test complete model creation workflow with mocked components."""
        # Create PINN
        pinn = create_spectroscopy_pinn(architecture="standard")
        
        # Mock all DeepXDE components
        with patch('model_definition.dde.geometry.Rectangle') as mock_rect, \
             patch('model_definition.dde.nn.FNN') as mock_fnn, \
             patch('model_definition.dde.data.PDE') as mock_pde, \
             patch('model_definition.dde.data.DataSet') as mock_dataset, \
             patch('model_definition.dde.data.combine.CombinedData') as mock_combined, \
             patch('model_definition.dde.Model') as mock_model:
            
            # Setup mocks
            mock_rect.return_value = MagicMock()
            mock_net = MagicMock()
            mock_fnn.return_value = mock_net
            mock_pde.return_value = MagicMock()
            mock_dataset.return_value = MagicMock()
            mock_combined.return_value = MagicMock()
            mock_model.return_value = MagicMock()
            
            # Create experimental data
            X_exp = np.random.rand(100, 2)
            y_exp = np.random.rand(100, 1)
            
            # Execute complete workflow
            model = pinn.create_model(experimental_data=(X_exp, y_exp))
            
            # Verify all components were created
            mock_rect.assert_called_once()
            mock_fnn.assert_called_once()
            mock_pde.assert_called_once()
            mock_dataset.assert_called_once()
            mock_combined.assert_called_once()
            mock_model.assert_called_once()
            
            # Verify transformations were applied
            mock_net.apply_input_transform.assert_called_once()
            mock_net.apply_output_transform.assert_called_once()
            
            assert model is not None
    
    def test_model_summary_completeness(self):
        """Test that model summary contains all required information."""
        pinn = create_spectroscopy_pinn(
            wavelength_range=(250, 750),
            concentration_range=(5, 50),
            architecture="deep"
        )
        
        summary = pinn.get_model_summary()
        
        # Verify all required sections and their content
        required_keys = [
            ("architecture", ["layer_sizes", "activation", "kernel_initializer", "total_parameters"]),
            ("domain", ["wavelength_range", "concentration_range", "path_length"]),
            ("physics", ["pde_constraint", "boundary_conditions", "regularization"]),
            ("geometry", ["type", "normalized_domain"])
        ]
        
        for section, keys in required_keys:
            assert section in summary
            for key in keys:
                assert key in summary[section]
        
        # Verify specific values
        assert summary["architecture"]["layer_sizes"] == [2, 64, 128, 256, 256, 128, 64, 32, 1]
        assert summary["domain"]["wavelength_range"] == (250, 750)
        assert summary["domain"]["concentration_range"] == (5, 50)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
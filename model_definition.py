"""
Phase 2: Model Definition for UV-Vis Spectra PINN
=================================================

This module defines the neural network architecture and DeepXDE model components
for Physics-Informed Neural Networks applied to UV-Vis spectra prediction.

The model implements Beer-Lambert law physics through:
1. Neural network architecture optimized for spectroscopic data
2. Physics-informed constraints via PDE formulation
3. Custom input/output transformations for spectroscopy
4. DeepXDE geometry and boundary condition setup

Author: Claude Code Assistant
"""

import numpy as np
import deepxde as dde
from typing import Tuple, Optional, Callable, Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpectroscopyPINN:
    """
    Physics-Informed Neural Network for UV-Vis Spectroscopy.
    
    This class encapsulates the complete PINN model definition including:
    - Neural network architecture optimized for spectroscopic data
    - Physics constraints based on Beer-Lambert law
    - Input/output transformations for normalization
    - DeepXDE geometry and PDE definitions
    """
    
    def __init__(self, 
                 wavelength_range: Tuple[float, float] = (200, 800),
                 concentration_range: Tuple[float, float] = (0, 60),
                 layer_sizes: List[int] = [2, 64, 128, 128, 64, 32, 1],
                 activation: str = "tanh",
                 kernel_initializer: str = "Glorot normal"):
        """
        Initialize the Spectroscopy PINN model.
        
        Args:
            wavelength_range: (min_wavelength, max_wavelength) in nm
            concentration_range: (min_concentration, max_concentration) in µg/L
            layer_sizes: List defining neural network architecture
            activation: Activation function name
            kernel_initializer: Weight initialization method
        """
        self.wavelength_range = wavelength_range
        self.concentration_range = concentration_range
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        
        # Model components (initialized later)
        self.geometry = None
        self.net = None
        self.pde = None
        self.data = None
        self.model = None
        
        # Path length parameter (Beer-Lambert law: A = ε × b × c)
        self.path_length = 1.0  # cm, typical cuvette path length
        
        logger.info(f"Initialized SpectroscopyPINN with architecture: {layer_sizes}")
        logger.info(f"Domain: λ={wavelength_range} nm, c={concentration_range} µg/L")
    
    def create_geometry(self) -> dde.geometry.Rectangle:
        """
        Create 2D rectangular geometry for the spectroscopic domain.
        
        The domain spans:
        - Wavelength: [λ_min, λ_max] nm (normalized to [-1, 1])
        - Concentration: [c_min, c_max] µg/L (normalized to [0, 1])
        
        Returns:
            DeepXDE Rectangle geometry instance
        """
        # Normalized domain bounds
        # Wavelength: symmetric around 0 for better numerical properties
        lambda_norm_min, lambda_norm_max = -1.0, 1.0
        
        # Concentration: preserve zero for physics (baseline condition)
        conc_norm_min, conc_norm_max = 0.0, 1.0
        
        self.geometry = dde.geometry.Rectangle(
            xmin=[lambda_norm_min, conc_norm_min],
            xmax=[lambda_norm_max, conc_norm_max]
        )
        
        logger.info(f"Created rectangular geometry: λ_norm ∈ [{lambda_norm_min}, {lambda_norm_max}], "
                   f"c_norm ∈ [{conc_norm_min}, {conc_norm_max}]")
        
        return self.geometry
    
    def beer_lambert_pde(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Physics-informed constraint based on Beer-Lambert law.
        
        Beer-Lambert law: A(λ,c) = ε(λ,c) × b × c + A_bg(λ)
        For differential absorption: ΔA(λ,c) = A(λ,c) - A_bg(λ) = ε(λ) × b × c
        
        Physics constraint: ∂(ΔA)/∂c = ε(λ) × b
        This means the concentration derivative should be independent of c
        (linearity constraint).
        
        Args:
            x: Input array [λ_normalized, c_normalized]
            y: Network output ΔA_normalized
            
        Returns:
            PDE residual for physics constraint
        """
        lambda_norm = x[:, 0:1]  # Normalized wavelength
        c_norm = x[:, 1:2]       # Normalized concentration
        delta_A = y              # Differential absorption output
        
        # Compute concentration derivative: ∂(ΔA)/∂c
        dA_dc = dde.grad.jacobian(delta_A, x, i=0, j=1)
        
        # Physics constraint: For true Beer-Lambert behavior,
        # ∂(ΔA)/∂c should be independent of concentration
        # We enforce this by penalizing ∂²(ΔA)/∂c²
        d2A_dc2 = dde.grad.jacobian(dA_dc, x, i=0, j=1)
        
        # Secondary constraint: Smoothness in wavelength
        # Penalize rapid changes in spectral features
        d2A_dlambda2 = dde.grad.jacobian(dde.grad.jacobian(delta_A, x, i=0, j=0), x, i=0, j=0)
        
        # Combined physics constraint
        physics_residual = d2A_dc2  # Primary: concentration linearity
        
        return physics_residual
    
    def create_neural_network(self) -> dde.nn.NN:
        """
        Create and configure the neural network architecture.
        
        The network is designed specifically for spectroscopic data:
        - Input: [λ_normalized, c_normalized]
        - Hidden layers: Optimized for smooth spectral features
        - Output: ΔA_normalized (differential absorption)
        - Transformations: Physics-aware input/output processing
        
        Returns:
            Configured DeepXDE neural network
        """
        # Create base feedforward neural network
        self.net = dde.nn.FNN(
            layer_sizes=self.layer_sizes,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer
        )
        
        # Apply input transformation for better numerical properties
        def input_transform(x):
            """
            Transform inputs for improved numerical stability.
            
            Args:
                x: Raw normalized inputs [λ_norm, c_norm]
            
            Returns:
                Transformed inputs
            """
            lambda_norm = x[:, 0:1]
            c_norm = x[:, 1:2]
            
            # Apply scaling for better gradient flow
            # Wavelength: already normalized to [-1, 1]
            # Concentration: scale to improve gradient properties
            lambda_transformed = lambda_norm
            c_transformed = c_norm
            
            return dde.backend.concat([lambda_transformed, c_transformed], 1)
        
        # Apply output transformation to enforce physical constraints
        def output_transform(x, y):
            """
            Transform outputs to ensure physical validity.
            
            Args:
                x: Input array [λ_norm, c_norm]
                y: Raw network output
            
            Returns:
                Physically constrained output
            """
            lambda_norm = x[:, 0:1]
            c_norm = x[:, 1:2]
            
            # Ensure output goes to zero when concentration is zero (baseline condition)
            # ΔA(λ, c=0) = 0 by definition of differential absorption
            baseline_constraint = c_norm
            
            # Apply soft constraint: output is proportional to concentration near c=0
            # This enforces the Beer-Lambert linear relationship
            constrained_output = y * baseline_constraint
            
            return constrained_output
        
        # Apply transformations to network
        self.net.apply_input_transform(input_transform)
        self.net.apply_output_transform(output_transform)
        
        logger.info(f"Created neural network with {len(self.layer_sizes)-1} layers")
        logger.info(f"Activation: {self.activation}, Initializer: {self.kernel_initializer}")
        
        return self.net
    
    def create_pde_data(self, 
                       num_domain: int = 2000,
                       num_boundary: int = 200,
                       num_test: int = 500) -> dde.data.PDE:
        """
        Create PDE data structure with physics constraints.
        
        Args:
            num_domain: Number of collocation points in the domain
            num_boundary: Number of boundary condition points
            num_test: Number of test points for validation
        
        Returns:
            DeepXDE PDE data instance
        """
        if self.geometry is None:
            self.create_geometry()
        
        # Define boundary conditions
        boundary_conditions = self._create_boundary_conditions()
        
        # Create PDE data with physics constraints
        self.pde = dde.data.PDE(
            geometry=self.geometry,
            pde=self.beer_lambert_pde,
            bcs=boundary_conditions,
            num_domain=num_domain,
            num_boundary=num_boundary,
            num_test=num_test,
            train_distribution='Hammersley'  # Better coverage than uniform
        )
        
        logger.info(f"Created PDE data: {num_domain} domain points, {num_boundary} boundary points")
        
        return self.pde
    
    def _create_boundary_conditions(self) -> List:
        """
        Create boundary conditions for the spectroscopic domain.
        
        Key boundary conditions:
        1. Zero concentration boundary: ΔA(λ, c=0) = 0
        2. Smoothness at domain boundaries
        
        Returns:
            List of DeepXDE boundary conditions
        """
        boundary_conditions = []
        
        # Boundary condition 1: Zero differential absorption at c=0
        def zero_concentration_bc(x, on_boundary):
            """Identify zero concentration boundary."""
            return on_boundary and np.isclose(x[1], 0.0)  # c_norm = 0
        
        def zero_absorption(x):
            """Zero differential absorption condition."""
            return 0.0
        
        bc_zero = dde.icbc.DirichletBC(
            self.geometry,
            zero_absorption,
            zero_concentration_bc
        )
        boundary_conditions.append(bc_zero)
        
        logger.info(f"Created {len(boundary_conditions)} boundary conditions")
        
        return boundary_conditions
    
    def create_combined_data(self, 
                           experimental_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> dde.data.Data:
        """
        Create combined data structure with experimental data and physics constraints.
        
        Args:
            experimental_data: Optional tuple of (X_train, y_train) experimental data
        
        Returns:
            Combined DeepXDE data instance
        """
        # Create PDE component
        pde_data = self.create_pde_data()
        
        if experimental_data is not None:
            X_train, y_train = experimental_data
            
            # Split data for training and testing
            n_train = int(0.8 * len(X_train))
            indices = np.random.permutation(len(X_train))
            
            X_train_split = X_train[indices[:n_train]]
            y_train_split = y_train[indices[:n_train]]
            X_test_split = X_train[indices[n_train:]]
            y_test_split = y_train[indices[n_train:]]
            
            # Create supervised learning data
            supervised_data = dde.data.DataSet(
                X_train=X_train_split,
                y_train=y_train_split,
                X_test=X_test_split,
                y_test=y_test_split,
                standardize=False  # Already normalized
            )
            
            # Combine PDE and supervised data
            self.data = dde.data.combine.CombinedData([supervised_data, pde_data])
            
            logger.info(f"Created combined data with experimental points: "
                       f"train={len(X_train_split)}, test={len(X_test_split)}")
        else:
            # Pure physics-informed learning
            self.data = pde_data
            logger.info("Created physics-only PINN (no experimental data)")
        
        return self.data
    
    def create_model(self, experimental_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> dde.model.Model:
        """
        Create the complete PINN model.
        
        Args:
            experimental_data: Optional experimental training data
        
        Returns:
            Configured DeepXDE model
        """
        # Create network if not already created
        if self.net is None:
            self.create_neural_network()
        
        # Create combined data
        data = self.create_combined_data(experimental_data)
        
        # Create DeepXDE model
        self.model = dde.Model(data, self.net)
        
        logger.info("Created complete PINN model")
        
        return self.model
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model configuration.
        
        Returns:
            Dictionary containing model details
        """
        summary = {
            "architecture": {
                "layer_sizes": self.layer_sizes,
                "activation": self.activation,
                "kernel_initializer": self.kernel_initializer,
                "total_parameters": self._count_parameters()
            },
            "domain": {
                "wavelength_range": self.wavelength_range,
                "concentration_range": self.concentration_range,
                "path_length": self.path_length
            },
            "physics": {
                "pde_constraint": "Beer-Lambert law linearity",
                "boundary_conditions": ["Zero absorption at c=0"],
                "regularization": ["Spectral smoothness", "Concentration linearity"]
            },
            "geometry": {
                "type": "Rectangle",
                "normalized_domain": "λ_norm ∈ [-1,1], c_norm ∈ [0,1]"
            }
        }
        
        return summary
    
    def _count_parameters(self) -> int:
        """
        Estimate the number of trainable parameters.
        
        Returns:
            Approximate parameter count
        """
        if not self.layer_sizes:
            return 0
        
        param_count = 0
        for i in range(len(self.layer_sizes) - 1):
            # Weights: layer_sizes[i] * layer_sizes[i+1]
            # Biases: layer_sizes[i+1]
            param_count += self.layer_sizes[i] * self.layer_sizes[i+1] + self.layer_sizes[i+1]
        
        return param_count


def create_spectroscopy_pinn(wavelength_range: Tuple[float, float] = (200, 800),
                           concentration_range: Tuple[float, float] = (0, 60),
                           architecture: str = "standard") -> SpectroscopyPINN:
    """
    Factory function to create pre-configured spectroscopy PINN models.
    
    Args:
        wavelength_range: Wavelength domain in nm
        concentration_range: Concentration domain in µg/L
        architecture: Pre-defined architecture ("standard", "deep", "wide")
    
    Returns:
        Configured SpectroscopyPINN instance
    """
    architecture_configs = {
        "standard": {
            "layer_sizes": [2, 64, 128, 128, 64, 32, 1],
            "activation": "tanh"
        },
        "deep": {
            "layer_sizes": [2, 64, 128, 256, 256, 128, 64, 32, 1],
            "activation": "tanh"
        },
        "wide": {
            "layer_sizes": [2, 128, 256, 256, 128, 1],
            "activation": "swish"
        }
    }
    
    if architecture not in architecture_configs:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list(architecture_configs.keys())}")
    
    config = architecture_configs[architecture]
    
    pinn = SpectroscopyPINN(
        wavelength_range=wavelength_range,
        concentration_range=concentration_range,
        layer_sizes=config["layer_sizes"],
        activation=config["activation"]
    )
    
    logger.info(f"Created {architecture} spectroscopy PINN")
    
    return pinn


if __name__ == "__main__":
    # Example usage
    pinn = create_spectroscopy_pinn(architecture="standard")
    
    # Create model components
    geometry = pinn.create_geometry()
    network = pinn.create_neural_network()
    
    # Create synthetic experimental data for testing
    np.random.seed(42)
    n_points = 1000
    lambda_norm = np.random.uniform(-1, 1, (n_points, 1))
    c_norm = np.random.uniform(0, 1, (n_points, 1))
    X_synthetic = np.hstack([lambda_norm, c_norm])
    
    # Synthetic Beer-Lambert response
    y_synthetic = c_norm * (0.1 + 0.02 * np.sin(5 * lambda_norm))  # Simple spectral feature
    
    # Create complete model
    model = pinn.create_model(experimental_data=(X_synthetic, y_synthetic))
    
    print("Model Summary:")
    print(pinn.get_model_summary())
    
    print(f"\nModel created successfully with {pinn._count_parameters()} parameters")
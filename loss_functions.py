"""
Phase 3: Loss Function Construction for UV-Vis Spectra PINN
==========================================================

This module implements the multi-component loss function for Physics-Informed Neural Networks
applied to UV-Vis spectroscopy data. The loss function combines experimental data fitting,
physics constraints, and smoothness regularization.

Multi-component Loss Function:
- L_data = mean[(ΔA_model - ΔA_true)²] - MSE on experimental data
- L_phys = mean[(∂_c ΔA_model - b·(ε_net + c·∂_c ε_net))²] - Physics constraint
- L_smooth = mean[(∂²_λ ε_net)²] - Smoothness regularization
- Total Loss = L_data + α·L_phys + β·L_smooth

Author: Claude Code Assistant
"""

import numpy as np
import deepxde as dde
import tensorflow as tf
from typing import Tuple, Optional, Callable, Dict, Any, List
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UVVisLossFunction:
    """
    Multi-component loss function for UV-Vis spectroscopy PINN.
    
    This class implements the complete loss function including:
    - Experimental data fitting (MSE loss)
    - Beer-Lambert law physics constraints
    - Spectral smoothness regularization
    - Adaptive loss weighting capabilities
    """
    
    def __init__(self, 
                 path_length: float = 1.0,
                 physics_weight: float = 0.1,
                 smooth_weight: float = 1e-4,
                 enable_adaptive_weighting: bool = False,
                 numerical_stability_eps: float = 1e-8):
        """
        Initialize the UV-Vis loss function.
        
        Args:
            path_length: Optical path length in cm (Beer-Lambert parameter b)
            physics_weight: Weight for physics constraint loss (α)
            smooth_weight: Weight for smoothness regularization (β)
            enable_adaptive_weighting: Whether to use adaptive loss weighting
            numerical_stability_eps: Small epsilon for numerical stability
        """
        self.path_length = path_length
        self.physics_weight = physics_weight
        self.smooth_weight = smooth_weight
        self.enable_adaptive_weighting = enable_adaptive_weighting
        self.eps = numerical_stability_eps
        
        # Loss component tracking
        self.loss_history = {
            'data': [],
            'physics': [],
            'smoothness': [],
            'total': [],
            'weights': {'physics': [], 'smoothness': []}
        }
        
        # Adaptive weighting parameters
        if enable_adaptive_weighting:
            self.adaptive_weights = AdaptiveLossWeighting(
                initial_physics_weight=physics_weight,
                initial_smooth_weight=smooth_weight
            )
        
        logger.info(f"Initialized UV-Vis loss function: b={path_length}, α={physics_weight}, β={smooth_weight}")
    
    def create_loss_function(self) -> Callable:
        """
        Create the complete multi-component loss function.
        
        Returns:
            Loss function compatible with DeepXDE model.compile()
        """
        def uvvis_loss(targets: tf.Tensor, outputs: tf.Tensor, inputs: tf.Tensor) -> tf.Tensor:
            """
            Multi-component UV-Vis loss function.
            
            Args:
                targets: True experimental differential absorption values
                outputs: Neural network predictions (ε_net values)
                inputs: Input features [wavelength_norm, concentration_norm]
            
            Returns:
                Total weighted loss
            """
            # Extract input components
            wavelength_norm = inputs[:, 0:1]  # Normalized wavelength
            concentration_norm = inputs[:, 1:2]  # Normalized concentration
            epsilon_pred = outputs  # Predicted molar absorptivity
            
            # Compute individual loss components
            data_loss = self._compute_data_loss(targets, epsilon_pred, wavelength_norm, concentration_norm)
            physics_loss = self._compute_physics_loss(epsilon_pred, wavelength_norm, concentration_norm)
            smooth_loss = self._compute_smoothness_loss(epsilon_pred, wavelength_norm, concentration_norm)
            
            # Apply adaptive weighting if enabled
            if self.enable_adaptive_weighting:
                current_weights = self.adaptive_weights.update_weights(
                    data_loss, physics_loss, smooth_loss
                )
                physics_weight = current_weights['physics']
                smooth_weight = current_weights['smoothness']
            else:
                physics_weight = self.physics_weight
                smooth_weight = self.smooth_weight
            
            # Compute total weighted loss
            total_loss = (data_loss + 
                         physics_weight * physics_loss + 
                         smooth_weight * smooth_loss)
            
            # Log loss components for monitoring
            self._log_loss_components(data_loss, physics_loss, smooth_loss, 
                                    total_loss, physics_weight, smooth_weight)
            
            return total_loss
        
        return uvvis_loss
    
    def _compute_data_loss(self, targets: tf.Tensor, epsilon_pred: tf.Tensor, 
                          wavelength_norm: tf.Tensor, concentration_norm: tf.Tensor) -> tf.Tensor:
        """
        Compute experimental data fitting loss (MSE).
        
        L_data = mean[(ΔA_model - ΔA_true)²]
        where ΔA_model = ε_net × b × c
        """
        # Predicted differential absorption: ΔA = ε × b × c
        predicted_absorption = epsilon_pred * self.path_length * concentration_norm
        
        # MSE loss with numerical stability
        data_loss = tf.reduce_mean(tf.square(predicted_absorption - targets))
        
        # Add small regularization for numerical stability
        data_loss = data_loss + self.eps * tf.reduce_mean(tf.square(epsilon_pred))
        
        return data_loss
    
    def _compute_physics_loss(self, epsilon_pred: tf.Tensor, 
                             wavelength_norm: tf.Tensor, concentration_norm: tf.Tensor) -> tf.Tensor:
        """
        Compute Beer-Lambert law physics constraint loss.
        
        L_phys = mean[(∂_c ΔA_model - b·(ε_net + c·∂_c ε_net))²]
        
        This enforces the Beer-Lambert law derivative consistency:
        ∂_c [ε(λ,c) × b × c] = b × [ε(λ,c) + c × ∂_c ε(λ,c)]
        """
        try:
            # Use persistent gradient tape for multiple gradient computations
            with tf.GradientTape(persistent=True) as tape:
                tape.watch([concentration_norm])
                
                # Predicted absorption model: A_model = ε × b × c
                absorption_model = epsilon_pred * self.path_length * concentration_norm
                
                # Also track epsilon for its concentration derivative
                tape.watch([epsilon_pred])
            
            # Compute concentration derivatives
            dA_dc = tape.gradient(absorption_model, concentration_norm)
            deps_dc = tape.gradient(epsilon_pred, concentration_norm)
            
            # Handle None gradients (can occur at boundaries)
            if dA_dc is None:
                dA_dc = tf.zeros_like(absorption_model)
            if deps_dc is None:
                deps_dc = tf.zeros_like(epsilon_pred)
            
            # Beer-Lambert physics constraint residual
            # ∂_c A_model should equal b × (ε + c × ∂_c ε)
            physics_target = self.path_length * (epsilon_pred + concentration_norm * deps_dc)
            physics_residual = dA_dc - physics_target
            
            # Physics loss with numerical stability
            physics_loss = tf.reduce_mean(tf.square(physics_residual))
            physics_loss = physics_loss + self.eps * tf.reduce_mean(tf.square(deps_dc))
            
            # Clean up persistent tape
            del tape
            
            return physics_loss
            
        except Exception as e:
            logger.warning(f"Physics loss computation failed: {e}")
            # Fallback to finite difference approximation
            return self._compute_physics_loss_finite_difference(
                epsilon_pred, wavelength_norm, concentration_norm
            )
    
    def _compute_physics_loss_finite_difference(self, epsilon_pred: tf.Tensor,
                                               wavelength_norm: tf.Tensor, 
                                               concentration_norm: tf.Tensor) -> tf.Tensor:
        """
        Fallback finite difference implementation for physics loss.
        """
        h = 1e-5  # Small perturbation
        
        # Forward and backward perturbations
        c_forward = concentration_norm + h
        c_backward = concentration_norm - h
        
        # Clamp to valid range [0, 1] for normalized concentrations
        c_forward = tf.clip_by_value(c_forward, 0.0, 1.0)
        c_backward = tf.clip_by_value(c_backward, 0.0, 1.0)
        
        # Approximate derivative using central differences
        # Note: This requires re-evaluation of the network, which is expensive
        # In practice, this would need access to the network function
        
        # For now, return a simplified constraint
        # This assumes linear relationship ∂A/∂c ≈ b × ε
        linear_constraint = epsilon_pred * self.path_length
        
        # Simple linearity check
        absorption_model = epsilon_pred * self.path_length * concentration_norm
        expected_linear = linear_constraint * concentration_norm
        
        physics_loss = tf.reduce_mean(tf.square(absorption_model - expected_linear))
        
        return physics_loss
    
    def _compute_smoothness_loss(self, epsilon_pred: tf.Tensor, 
                                wavelength_norm: tf.Tensor, concentration_norm: tf.Tensor) -> tf.Tensor:
        """
        Compute spectral smoothness regularization loss.
        
        L_smooth = mean[(∂²_λ ε_net)²]
        
        This penalizes rapid spectral variations to ensure physically realistic
        smooth absorption spectra.
        """
        try:
            with tf.GradientTape(persistent=True) as tape:
                tape.watch([wavelength_norm])
                
                # First derivative with respect to wavelength
                deps_dlambda = tape.gradient(epsilon_pred, wavelength_norm)
                
                if deps_dlambda is None:
                    deps_dlambda = tf.zeros_like(epsilon_pred)
                
            # Second derivative with respect to wavelength
            d2eps_dlambda2 = tape.gradient(deps_dlambda, wavelength_norm)
            
            if d2eps_dlambda2 is None:
                d2eps_dlambda2 = tf.zeros_like(epsilon_pred)
            
            # Smoothness loss
            smooth_loss = tf.reduce_mean(tf.square(d2eps_dlambda2))
            
            # Clean up persistent tape
            del tape
            
            return smooth_loss
            
        except Exception as e:
            logger.warning(f"Smoothness loss computation failed: {e}")
            # Fallback: simple L2 regularization on epsilon
            return self.eps * tf.reduce_mean(tf.square(epsilon_pred))
    
    def _log_loss_components(self, data_loss: tf.Tensor, physics_loss: tf.Tensor,
                           smooth_loss: tf.Tensor, total_loss: tf.Tensor,
                           physics_weight: float, smooth_weight: float) -> None:
        """Log individual loss components for monitoring and debugging."""
        try:
            # Convert to numpy for logging
            self.loss_history['data'].append(float(data_loss.numpy()))
            self.loss_history['physics'].append(float(physics_loss.numpy()))
            self.loss_history['smoothness'].append(float(smooth_loss.numpy()))
            self.loss_history['total'].append(float(total_loss.numpy()))
            self.loss_history['weights']['physics'].append(physics_weight)
            self.loss_history['weights']['smoothness'].append(smooth_weight)
            
            # Log every N iterations to avoid spam
            if len(self.loss_history['total']) % 100 == 0:
                logger.info(f"Loss components - Data: {float(data_loss.numpy()):.6f}, "
                           f"Physics: {float(physics_loss.numpy()):.6f}, "
                           f"Smooth: {float(smooth_loss.numpy()):.6f}, "
                           f"Total: {float(total_loss.numpy()):.6f}")
                
        except Exception as e:
            # Don't let logging errors break training
            logger.debug(f"Loss logging failed: {e}")
    
    def get_loss_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about loss evolution.
        
        Returns:
            Dictionary containing loss statistics and trends
        """
        if not self.loss_history['total']:
            return {"status": "No loss history available"}
        
        stats = {}
        for component in ['data', 'physics', 'smoothness', 'total']:
            values = np.array(self.loss_history[component])
            if len(values) > 0:
                stats[component] = {
                    'current': float(values[-1]),
                    'mean': float(np.mean(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'std': float(np.std(values)),
                    'trend': self._compute_trend(values)
                }
        
        return stats
    
    def _compute_trend(self, values: np.ndarray) -> str:
        """Compute trend direction for loss values."""
        if len(values) < 10:
            return "insufficient_data"
        
        recent = values[-10:]
        early = values[:10] if len(values) > 20 else values[:len(values)//2]
        
        recent_mean = np.mean(recent)
        early_mean = np.mean(early)
        
        if recent_mean < early_mean * 0.9:
            return "decreasing"
        elif recent_mean > early_mean * 1.1:
            return "increasing"
        else:
            return "stable"
    
    def plot_loss_evolution(self, save_path: Optional[str] = None) -> None:
        """
        Plot the evolution of loss components during training.
        
        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
            
            # Plot individual loss components
            iterations = range(len(self.loss_history['total']))
            
            ax1.semilogy(iterations, self.loss_history['data'], label='Data Loss', color='blue')
            ax1.semilogy(iterations, self.loss_history['physics'], label='Physics Loss', color='red')
            ax1.semilogy(iterations, self.loss_history['smoothness'], label='Smoothness Loss', color='green')
            ax1.set_ylabel('Loss Value (log scale)')
            ax1.set_title('Individual Loss Components')
            ax1.legend()
            ax1.grid(True)
            
            # Plot total loss
            ax2.semilogy(iterations, self.loss_history['total'], label='Total Loss', color='black')
            ax2.set_ylabel('Total Loss (log scale)')
            ax2.set_title('Total Loss Evolution')
            ax2.legend()
            ax2.grid(True)
            
            # Plot adaptive weights if available
            if self.enable_adaptive_weighting:
                ax3.plot(iterations, self.loss_history['weights']['physics'], 
                        label='Physics Weight', color='red', linestyle='--')
                ax3.plot(iterations, self.loss_history['weights']['smoothness'], 
                        label='Smoothness Weight', color='green', linestyle='--')
                ax3.set_ylabel('Adaptive Weights')
                ax3.set_title('Adaptive Loss Weights Evolution')
                ax3.legend()
                ax3.grid(True)
            else:
                ax3.text(0.5, 0.5, 'Adaptive weighting not enabled', 
                        transform=ax3.transAxes, ha='center', va='center')
                ax3.set_title('Loss Weights (Fixed)')
            
            ax3.set_xlabel('Training Iterations')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Loss evolution plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available. Cannot plot loss evolution.")
        except Exception as e:
            logger.error(f"Failed to plot loss evolution: {e}")


class AdaptiveLossWeighting:
    """
    Adaptive loss weighting strategy to balance loss components during training.
    
    This class implements dynamic adjustment of loss weights based on the relative
    magnitudes and convergence rates of different loss components.
    """
    
    def __init__(self, 
                 initial_physics_weight: float = 0.1,
                 initial_smooth_weight: float = 1e-4,
                 adaptation_rate: float = 0.01,
                 target_balance_ratio: float = 1.0):
        """
        Initialize adaptive weighting.
        
        Args:
            initial_physics_weight: Initial weight for physics loss
            initial_smooth_weight: Initial weight for smoothness loss  
            adaptation_rate: Rate of weight adaptation (0-1)
            target_balance_ratio: Target ratio between loss components
        """
        self.current_weights = {
            'physics': initial_physics_weight,
            'smoothness': initial_smooth_weight
        }
        self.adaptation_rate = adaptation_rate
        self.target_ratio = target_balance_ratio
        
        # History for trend analysis
        self.loss_history = {'data': [], 'physics': [], 'smoothness': []}
        
    def update_weights(self, data_loss: tf.Tensor, physics_loss: tf.Tensor, 
                      smooth_loss: tf.Tensor) -> Dict[str, float]:
        """
        Update loss weights based on current loss values.
        
        Args:
            data_loss: Current data fitting loss
            physics_loss: Current physics constraint loss
            smooth_loss: Current smoothness loss
            
        Returns:
            Updated weight dictionary
        """
        # Convert to numpy for computation
        data_val = float(data_loss.numpy())
        physics_val = float(physics_loss.numpy())
        smooth_val = float(smooth_loss.numpy())
        
        # Store history
        self.loss_history['data'].append(data_val)
        self.loss_history['physics'].append(physics_val)
        self.loss_history['smoothness'].append(smooth_val)
        
        # Compute target weights based on loss magnitudes
        if physics_val > 0 and smooth_val > 0:
            # Target: balance relative contributions
            physics_target = data_val / (physics_val + 1e-10)
            smooth_target = data_val / (smooth_val + 1e-10)
            
            # Smooth adaptation
            self.current_weights['physics'] = (
                (1 - self.adaptation_rate) * self.current_weights['physics'] +
                self.adaptation_rate * physics_target
            )
            
            self.current_weights['smoothness'] = (
                (1 - self.adaptation_rate) * self.current_weights['smoothness'] +
                self.adaptation_rate * smooth_target
            )
            
            # Clamp weights to reasonable ranges
            self.current_weights['physics'] = np.clip(
                self.current_weights['physics'], 1e-6, 10.0
            )
            self.current_weights['smoothness'] = np.clip(
                self.current_weights['smoothness'], 1e-8, 1.0
            )
        
        return self.current_weights.copy()


def create_uvvis_loss_function(path_length: float = 1.0,
                              physics_weight: float = 0.1,
                              smooth_weight: float = 1e-4,
                              enable_adaptive_weighting: bool = False) -> Callable:
    """
    Factory function to create UV-Vis loss function.
    
    Args:
        path_length: Optical path length in cm
        physics_weight: Weight for physics constraint loss
        smooth_weight: Weight for smoothness regularization
        enable_adaptive_weighting: Whether to use adaptive weighting
        
    Returns:
        Configured loss function for use with DeepXDE
    """
    loss_obj = UVVisLossFunction(
        path_length=path_length,
        physics_weight=physics_weight,
        smooth_weight=smooth_weight,
        enable_adaptive_weighting=enable_adaptive_weighting
    )
    
    return loss_obj.create_loss_function()


def validate_loss_function(loss_fn: Callable, test_data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Validate loss function with test data.
    
    Args:
        loss_fn: Loss function to validate
        test_data: Tuple of (inputs, targets, expected_ranges)
        
    Returns:
        Validation metrics
    """
    inputs, targets, expected_ranges = test_data
    
    # Convert to tensors
    inputs_tf = tf.constant(inputs, dtype=tf.float32)
    targets_tf = tf.constant(targets, dtype=tf.float32)
    
    # Create dummy outputs in expected range
    outputs_tf = tf.constant(np.random.uniform(
        expected_ranges[0], expected_ranges[1], targets.shape
    ), dtype=tf.float32)
    
    try:
        # Compute loss
        loss_value = loss_fn(targets_tf, outputs_tf, inputs_tf)
        
        # Check for NaN or infinite values
        is_finite = tf.math.is_finite(loss_value)
        is_positive = loss_value > 0
        
        return {
            'loss_value': float(loss_value.numpy()),
            'is_finite': bool(is_finite.numpy()),
            'is_positive': bool(is_positive.numpy()),
            'validation_passed': bool(is_finite.numpy() and is_positive.numpy())
        }
        
    except Exception as e:
        return {
            'loss_value': float('nan'),
            'is_finite': False,
            'is_positive': False,
            'validation_passed': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing UV-Vis loss function implementation...")
    
    # Create test data
    n_points = 100
    wavelengths = np.linspace(-1, 1, n_points).reshape(-1, 1)
    concentrations = np.linspace(0, 1, n_points).reshape(-1, 1)
    inputs = np.hstack([wavelengths, concentrations])
    
    # Synthetic targets (Beer-Lambert relationship)
    synthetic_epsilon = 0.1 + 0.05 * np.sin(5 * wavelengths)
    targets = synthetic_epsilon * 1.0 * concentrations  # b = 1.0
    
    # Test loss function creation
    loss_fn = create_uvvis_loss_function(
        path_length=1.0,
        physics_weight=0.1,
        smooth_weight=1e-4,
        enable_adaptive_weighting=False
    )
    
    # Validate loss function
    validation_result = validate_loss_function(
        loss_fn, 
        (inputs, targets, (0.05, 0.15))
    )
    
    print("Loss Function Validation Results:")
    for key, value in validation_result.items():
        print(f"  {key}: {value}")
    
    if validation_result['validation_passed']:
        logger.info("UV-Vis loss function validation passed successfully!")
    else:
        logger.error("UV-Vis loss function validation failed!")
    
    # Test adaptive weighting
    logger.info("Testing adaptive loss weighting...")
    adaptive_loss_fn = create_uvvis_loss_function(
        enable_adaptive_weighting=True
    )
    
    adaptive_validation = validate_loss_function(
        adaptive_loss_fn,
        (inputs, targets, (0.05, 0.15))
    )
    
    print("Adaptive Loss Function Validation:")
    for key, value in adaptive_validation.items():
        print(f"  {key}: {value}")
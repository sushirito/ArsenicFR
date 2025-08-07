"""
Phase 1: Data Loading and Preprocessing for UV-Vis Spectra PINN
================================================================

This module handles loading UV-Vis spectroscopic data from CSV files and preprocessing
it for use with DeepXDE-based Physics-Informed Neural Networks (PINNs).

The preprocessing implements Beer-Lambert law physics by:
1. Loading experimental data from CSV with validation
2. Extracting baseline absorption (c=0 condition)  
3. Computing concentration-dependent differential absorption
4. Normalizing inputs for neural network training
5. Creating DeepXDE-compatible data structures

Author: Claude Code Assistant
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UVVisDataProcessor:
    """
    Processes UV-Vis spectroscopic data for Physics-Informed Neural Networks.
    
    This class handles the complete data preprocessing pipeline from raw CSV data
    to normalized, physics-aware data structures compatible with DeepXDE.
    """
    
    def __init__(self, csv_path: str, validate_data: bool = True):
        """
        Initialize the UV-Vis data processor.
        
        Args:
            csv_path: Path to the CSV file containing UV-Vis data
            validate_data: Whether to perform extensive data validation
        """
        self.csv_path = Path(csv_path)
        self.validate_data = validate_data
        
        # Data storage
        self.raw_data: Optional[pd.DataFrame] = None
        self.wavelengths: Optional[np.ndarray] = None
        self.concentrations: Optional[np.ndarray] = None
        self.absorbance_matrix: Optional[np.ndarray] = None
        self.baseline_absorption: Optional[np.ndarray] = None
        self.differential_absorption: Optional[np.ndarray] = None
        
        # Normalization parameters
        self.wavelength_norm_params: Dict[str, float] = {}
        self.concentration_norm_params: Dict[str, float] = {}
        self.absorbance_norm_params: Dict[str, float] = {}
        
        logger.info(f"Initialized UV-Vis data processor for: {csv_path}")
    
    def load_and_validate_data(self) -> pd.DataFrame:
        """
        Load CSV data and perform validation checks.
        
        Returns:
            Loaded and validated DataFrame
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If data format is invalid
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        # Load CSV data
        try:
            self.raw_data = pd.read_csv(self.csv_path)
            logger.info(f"Loaded CSV with shape: {self.raw_data.shape}")
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}")
        
        if self.validate_data:
            self._validate_data_structure()
            self._validate_data_ranges()
        
        return self.raw_data
    
    def _validate_data_structure(self) -> None:
        """Validate the structure and format of the loaded data."""
        expected_columns = ['Wavelength', '0', '10', '20', '30', '40', '60']
        
        if list(self.raw_data.columns) != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {list(self.raw_data.columns)}")
        
        # Check for expected number of wavelength points (should be 601)
        expected_rows = 601
        if len(self.raw_data) != expected_rows:
            logger.warning(f"Expected {expected_rows} rows, got {len(self.raw_data)}")
        
        # Check for missing values
        if self.raw_data.isnull().any().any():
            raise ValueError("Data contains missing values")
        
        # Check data types
        for col in self.raw_data.columns:
            if not pd.api.types.is_numeric_dtype(self.raw_data[col]):
                raise ValueError(f"Column {col} contains non-numeric data")
        
        logger.info("Data structure validation passed")
    
    def _validate_data_ranges(self) -> None:
        """Validate that data values are within expected ranges."""
        wavelengths = self.raw_data['Wavelength'].values
        
        # Check wavelength range (should be 200-800 nm, descending)
        if wavelengths[0] < wavelengths[-1]:
            logger.warning("Wavelengths appear to be in ascending order, expected descending")
        
        min_wl, max_wl = wavelengths.min(), wavelengths.max()
        if not (190 <= min_wl <= 210) or not (790 <= max_wl <= 810):
            logger.warning(f"Wavelength range {min_wl}-{max_wl} nm may be outside expected 200-800 nm")
        
        # Check absorbance values (should be positive, typically 0-3 range)
        concentration_cols = ['0', '10', '20', '30', '40', '60']
        for col in concentration_cols:
            abs_values = self.raw_data[col].values
            if np.any(abs_values < 0):
                logger.warning(f"Negative absorbance values found in column {col}")
            if np.any(abs_values > 5):
                logger.warning(f"Unusually high absorbance values (>5) found in column {col}")
        
        logger.info("Data range validation completed")
    
    def extract_spectral_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract wavelengths, concentrations, and absorbance matrix from raw data.
        
        Returns:
            Tuple of (wavelengths, concentrations, absorbance_matrix)
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_and_validate_data() first.")
        
        # Extract wavelengths (convert to ascending order for physics consistency)
        self.wavelengths = self.raw_data['Wavelength'].values
        if self.wavelengths[0] > self.wavelengths[-1]:
            # Data is in descending order, reverse it
            self.wavelengths = self.wavelengths[::-1]
            reverse_data = True
        else:
            reverse_data = False
        
        # Extract concentrations (µg/L)
        self.concentrations = np.array([0, 10, 20, 30, 40, 60], dtype=np.float32)
        
        # Extract absorbance matrix [wavelengths × concentrations]
        concentration_cols = ['0', '10', '20', '30', '40', '60']
        absorbance_data = self.raw_data[concentration_cols].values
        
        if reverse_data:
            absorbance_data = absorbance_data[::-1, :]  # Reverse wavelength order
        
        self.absorbance_matrix = absorbance_data.astype(np.float32)
        
        logger.info(f"Extracted spectral components: {len(self.wavelengths)} wavelengths, "
                   f"{len(self.concentrations)} concentrations")
        
        return self.wavelengths, self.concentrations, self.absorbance_matrix
    
    def compute_beer_lambert_components(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Beer-Lambert law components: baseline and differential absorption.
        
        For Beer-Lambert law: A(λ,c) = ε(λ,c) × b × c + A_bg(λ)
        - A_bg(λ) = A(λ, c=0) is the baseline absorption
        - ΔA(λ,c) = A(λ,c) - A_bg(λ) is the concentration-dependent component
        
        Returns:
            Tuple of (baseline_absorption, differential_absorption)
        """
        if self.absorbance_matrix is None:
            raise ValueError("Spectral components not extracted. Call extract_spectral_components() first.")
        
        # Baseline absorption: A_bg(λ) = A(λ, c=0)
        self.baseline_absorption = self.absorbance_matrix[:, 0].copy()  # c=0 column
        
        # Differential absorption: ΔA(λ,c) = A(λ,c) - A_bg(λ) 
        # Only for non-zero concentrations
        differential_matrix = []
        concentrations_nonzero = []
        
        for i, conc in enumerate(self.concentrations):
            if conc > 0:  # Skip zero concentration
                diff_abs = self.absorbance_matrix[:, i] - self.baseline_absorption
                differential_matrix.append(diff_abs)
                concentrations_nonzero.append(conc)
        
        self.differential_absorption = np.column_stack(differential_matrix)
        self.concentrations_nonzero = np.array(concentrations_nonzero)
        
        logger.info(f"Computed Beer-Lambert components: baseline + {len(concentrations_nonzero)} "
                   f"concentration-dependent spectra")
        
        return self.baseline_absorption, self.differential_absorption
    
    def normalize_inputs(self) -> Dict[str, np.ndarray]:
        """
        Normalize inputs for neural network training.
        
        Normalization strategy:
        - Wavelength: λ_norm = (λ - λ_center) / λ_scale 
        - Concentration: c_norm = c / c_max (preserve zero)
        - Differential absorption: ΔA_norm = (ΔA - ΔA_min) / (ΔA_max - ΔA_min)
        
        Returns:
            Dictionary containing normalized data and parameters
        """
        if self.wavelengths is None or self.differential_absorption is None:
            raise ValueError("Components not computed. Call compute_beer_lambert_components() first.")
        
        # Wavelength normalization (center at middle of range)
        wavelength_min, wavelength_max = self.wavelengths.min(), self.wavelengths.max()
        wavelength_center = (wavelength_min + wavelength_max) / 2
        wavelength_scale = (wavelength_max - wavelength_min) / 2
        
        wavelengths_normalized = (self.wavelengths - wavelength_center) / wavelength_scale
        
        self.wavelength_norm_params = {
            'center': wavelength_center,
            'scale': wavelength_scale,
            'min': wavelength_min,
            'max': wavelength_max
        }
        
        # Concentration normalization (preserve zero, scale by maximum)
        concentration_max = self.concentrations_nonzero.max()
        concentrations_normalized = self.concentrations_nonzero / concentration_max
        
        self.concentration_norm_params = {
            'scale': concentration_max,
            'max': concentration_max,
            'min': 0.0
        }
        
        # Differential absorption normalization
        diff_abs_min = self.differential_absorption.min()
        diff_abs_max = self.differential_absorption.max()
        diff_abs_range = diff_abs_max - diff_abs_min
        
        if diff_abs_range > 0:
            differential_absorption_normalized = ((self.differential_absorption - diff_abs_min) / 
                                                 diff_abs_range)
        else:
            differential_absorption_normalized = self.differential_absorption
            logger.warning("Zero range in differential absorption, normalization skipped")
        
        self.absorbance_norm_params = {
            'min': diff_abs_min,
            'max': diff_abs_max,
            'range': diff_abs_range
        }
        
        normalized_data = {
            'wavelengths_norm': wavelengths_normalized,
            'concentrations_norm': concentrations_normalized,
            'differential_absorption_norm': differential_absorption_normalized,
            'wavelengths_raw': self.wavelengths,
            'concentrations_raw': self.concentrations_nonzero,
            'differential_absorption_raw': self.differential_absorption,
            'baseline_absorption': self.baseline_absorption
        }
        
        logger.info("Input normalization completed")
        return normalized_data
    
    def create_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training input-output pairs for neural network.
        
        Each training point consists of:
        - Input: [λ_normalized, c_normalized]
        - Output: [ΔA_normalized(λ,c)]
        
        Returns:
            Tuple of (X_train, y_train) arrays
        """
        normalized_data = self.normalize_inputs()
        
        wavelengths_norm = normalized_data['wavelengths_norm']
        concentrations_norm = normalized_data['concentrations_norm'] 
        differential_absorption_norm = normalized_data['differential_absorption_norm']
        
        # Create all combinations of (wavelength, concentration) pairs
        X_train_list = []
        y_train_list = []
        
        for i, wl_norm in enumerate(wavelengths_norm):
            for j, conc_norm in enumerate(concentrations_norm):
                # Input: [wavelength_normalized, concentration_normalized]
                x_point = np.array([wl_norm, conc_norm])
                X_train_list.append(x_point)
                
                # Output: differential absorption at this (λ, c) point
                y_point = differential_absorption_norm[i, j]
                y_train_list.append(y_point)
        
        X_train = np.array(X_train_list, dtype=np.float32)
        y_train = np.array(y_train_list, dtype=np.float32).reshape(-1, 1)
        
        logger.info(f"Created training data: {X_train.shape} inputs, {y_train.shape} outputs")
        return X_train, y_train
    
    def denormalize_wavelength(self, wavelength_norm: np.ndarray) -> np.ndarray:
        """Denormalize wavelength values."""
        return (wavelength_norm * self.wavelength_norm_params['scale'] + 
                self.wavelength_norm_params['center'])
    
    def denormalize_concentration(self, concentration_norm: np.ndarray) -> np.ndarray:
        """Denormalize concentration values.""" 
        return concentration_norm * self.concentration_norm_params['scale']
    
    def denormalize_absorption(self, absorption_norm: np.ndarray) -> np.ndarray:
        """Denormalize differential absorption values."""
        return (absorption_norm * self.absorbance_norm_params['range'] + 
                self.absorbance_norm_params['min'])
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the processed data.
        
        Returns:
            Dictionary containing data statistics and parameters
        """
        if self.raw_data is None:
            return {"status": "No data loaded"}
        
        summary = {
            "data_shape": self.raw_data.shape,
            "wavelength_range": (self.wavelengths.min(), self.wavelengths.max()) if self.wavelengths is not None else None,
            "concentrations": list(self.concentrations) if self.concentrations is not None else None,
            "baseline_stats": {
                "min": self.baseline_absorption.min(),
                "max": self.baseline_absorption.max(),
                "mean": self.baseline_absorption.mean()
            } if self.baseline_absorption is not None else None,
            "differential_absorption_stats": {
                "min": self.differential_absorption.min(),
                "max": self.differential_absorption.max(), 
                "mean": self.differential_absorption.mean()
            } if self.differential_absorption is not None else None,
            "normalization_params": {
                "wavelength": self.wavelength_norm_params,
                "concentration": self.concentration_norm_params,
                "absorbance": self.absorbance_norm_params
            }
        }
        
        return summary


def load_uvvis_data(csv_path: str, validate: bool = True) -> UVVisDataProcessor:
    """
    Convenience function to load and process UV-Vis data.
    
    Args:
        csv_path: Path to CSV file
        validate: Whether to perform data validation
        
    Returns:
        Configured UVVisDataProcessor instance
    """
    processor = UVVisDataProcessor(csv_path, validate_data=validate)
    processor.load_and_validate_data()
    processor.extract_spectral_components()
    processor.compute_beer_lambert_components()
    
    return processor


if __name__ == "__main__":
    # Example usage
    csv_path = "/Users/aditya/CodingProjects/datasets/0.30MB_AuNP_As.csv"
    
    try:
        processor = load_uvvis_data(csv_path)
        X_train, y_train = processor.create_training_data()
        
        print(f"Loaded UV-Vis data successfully:")
        print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Data summary: {processor.get_data_summary()}")
        
    except Exception as e:
        print(f"Error processing data: {e}")
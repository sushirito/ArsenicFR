"""
Unit Tests for Phase 1: Data Loading and Preprocessing
=====================================================

Comprehensive test suite for UV-Vis data preprocessing components.
Tests cover data loading, validation, normalization, and Beer-Lambert calculations.

Author: Claude Code Assistant
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from data_preprocessing import UVVisDataProcessor, load_uvvis_data


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    # Create realistic UV-Vis data
    wavelengths = np.linspace(800, 200, 601)  # Descending order as in real data
    concentrations = [0, 10, 20, 30, 40, 60]
    
    # Generate synthetic Beer-Lambert data
    data = {'Wavelength': wavelengths}
    baseline = 0.1 + 0.02 * np.exp(-(wavelengths - 300)**2 / 10000)  # Gaussian absorption
    
    for i, conc in enumerate(concentrations):
        if conc == 0:
            # Baseline absorption
            absorbance = baseline + np.random.normal(0, 0.001, len(wavelengths))
        else:
            # Beer-Lambert: A = baseline + ε × b × c
            epsilon = 0.001 + 0.0005 * np.exp(-(wavelengths - 400)**2 / 5000)
            absorbance = baseline + epsilon * 1.0 * conc + np.random.normal(0, 0.002, len(wavelengths))
        
        data[str(conc)] = absorbance
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_csv_file(sample_csv_data):
    """Create temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_csv_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


class TestUVVisDataProcessor:
    """Test suite for UVVisDataProcessor class."""
    
    @pytest.fixture 
    def processor(self, sample_csv_file):
        """Create UVVisDataProcessor instance with sample data."""
        return UVVisDataProcessor(sample_csv_file, validate_data=True)
    
    def test_initialization(self, sample_csv_file):
        """Test processor initialization."""
        processor = UVVisDataProcessor(sample_csv_file, validate_data=True)
        
        assert processor.csv_path == Path(sample_csv_file)
        assert processor.validate_data is True
        assert processor.raw_data is None
        assert processor.wavelengths is None
        assert processor.concentrations is None
    
    def test_load_and_validate_data_success(self, processor):
        """Test successful data loading and validation."""
        data = processor.load_and_validate_data()
        
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert data.shape[0] == 601  # Expected number of wavelengths
        assert list(data.columns) == ['Wavelength', '0', '10', '20', '30', '40', '60']
        assert processor.raw_data is not None
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises FileNotFoundError."""
        processor = UVVisDataProcessor("nonexistent_file.csv")
        
        with pytest.raises(FileNotFoundError):
            processor.load_and_validate_data()
    
    def test_validate_data_structure_invalid_columns(self, sample_csv_data):
        """Test validation fails with invalid column structure."""
        # Modify column names to be invalid
        invalid_data = sample_csv_data.copy()
        invalid_data.columns = ['Wave', '0', '10', '20', '30', '40', '60']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            invalid_data.to_csv(f.name, index=False)
            processor = UVVisDataProcessor(f.name, validate_data=True)
            
            with pytest.raises(ValueError, match="Expected columns"):
                processor.load_and_validate_data()
            
            os.unlink(f.name)
    
    def test_validate_data_structure_missing_values(self, sample_csv_data):
        """Test validation fails with missing values."""
        invalid_data = sample_csv_data.copy()
        invalid_data.iloc[100, 1] = np.nan  # Introduce NaN
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            invalid_data.to_csv(f.name, index=False)
            processor = UVVisDataProcessor(f.name, validate_data=True)
            
            with pytest.raises(ValueError, match="Data contains missing values"):
                processor.load_and_validate_data()
            
            os.unlink(f.name)
    
    def test_extract_spectral_components(self, processor):
        """Test extraction of wavelengths, concentrations, and absorbance matrix."""
        processor.load_and_validate_data()
        wavelengths, concentrations, absorbance_matrix = processor.extract_spectral_components()
        
        assert isinstance(wavelengths, np.ndarray)
        assert isinstance(concentrations, np.ndarray)
        assert isinstance(absorbance_matrix, np.ndarray)
        
        # Check dimensions
        assert len(wavelengths) == 601
        assert len(concentrations) == 6
        assert absorbance_matrix.shape == (601, 6)
        
        # Check wavelengths are in ascending order
        assert np.all(wavelengths[:-1] <= wavelengths[1:])
        
        # Check concentrations
        np.testing.assert_array_equal(concentrations, [0, 10, 20, 30, 40, 60])
    
    def test_compute_beer_lambert_components(self, processor):
        """Test Beer-Lambert baseline and differential calculations."""
        processor.load_and_validate_data()
        processor.extract_spectral_components()
        baseline, differential = processor.compute_beer_lambert_components()
        
        assert isinstance(baseline, np.ndarray)
        assert isinstance(differential, np.ndarray)
        
        # Check dimensions
        assert len(baseline) == 601  # One value per wavelength
        assert differential.shape == (601, 5)  # Exclude zero concentration
        
        # Baseline should be the zero concentration column
        np.testing.assert_array_equal(baseline, processor.absorbance_matrix[:, 0])
        
        # Check non-zero concentrations were processed
        assert len(processor.concentrations_nonzero) == 5
        np.testing.assert_array_equal(processor.concentrations_nonzero, [10, 20, 30, 40, 60])
    
    def test_normalize_inputs(self, processor):
        """Test input normalization for neural network."""
        processor.load_and_validate_data()
        processor.extract_spectral_components()
        processor.compute_beer_lambert_components()
        
        normalized_data = processor.normalize_inputs()
        
        # Check keys are present
        required_keys = [
            'wavelengths_norm', 'concentrations_norm', 'differential_absorption_norm',
            'wavelengths_raw', 'concentrations_raw', 'differential_absorption_raw',
            'baseline_absorption'
        ]
        for key in required_keys:
            assert key in normalized_data
        
        # Check normalization ranges
        wavelengths_norm = normalized_data['wavelengths_norm']
        concentrations_norm = normalized_data['concentrations_norm']
        diff_abs_norm = normalized_data['differential_absorption_norm']
        
        # Wavelengths should be centered and scaled
        assert wavelengths_norm.min() < -0.9 and wavelengths_norm.max() > 0.9
        
        # Concentrations should be scaled to [0, 1]
        assert concentrations_norm.min() >= 0
        assert concentrations_norm.max() <= 1
        
        # Differential absorption should be normalized to [0, 1] range
        assert diff_abs_norm.min() >= -0.1  # Small tolerance for numerical precision
        assert diff_abs_norm.max() <= 1.1
    
    def test_create_training_data(self, processor):
        """Test creation of training input-output pairs."""
        processor.load_and_validate_data()
        processor.extract_spectral_components()
        processor.compute_beer_lambert_components()
        
        X_train, y_train = processor.create_training_data()
        
        # Check dimensions
        n_wavelengths = 601
        n_concentrations = 5  # Non-zero concentrations only
        expected_points = n_wavelengths * n_concentrations
        
        assert X_train.shape == (expected_points, 2)  # [wavelength, concentration]
        assert y_train.shape == (expected_points, 1)  # differential absorption
        
        # Check data types
        assert X_train.dtype == np.float32
        assert y_train.dtype == np.float32
        
        # Check input ranges (normalized)
        assert X_train[:, 0].min() < -0.9  # Wavelength normalized
        assert X_train[:, 0].max() > 0.9
        assert X_train[:, 1].min() >= 0    # Concentration normalized
        assert X_train[:, 1].max() <= 1
    
    def test_denormalization_functions(self, processor):
        """Test denormalization functions for inverse transforms."""
        processor.load_and_validate_data()
        processor.extract_spectral_components()
        processor.compute_beer_lambert_components()
        processor.normalize_inputs()
        
        # Test wavelength denormalization
        wl_norm = np.array([-1, 0, 1])
        wl_denorm = processor.denormalize_wavelength(wl_norm)
        
        # Should recover original range approximately
        assert wl_denorm[0] < processor.wavelengths.min() + 10  # Some tolerance
        assert wl_denorm[-1] > processor.wavelengths.max() - 10
        
        # Test concentration denormalization
        c_norm = np.array([0, 0.5, 1])
        c_denorm = processor.denormalize_concentration(c_norm)
        
        assert c_denorm[0] == 0  # Zero should remain zero
        assert c_denorm[-1] == processor.concentrations_nonzero.max()
        
        # Test round-trip normalization/denormalization
        original_wl = processor.wavelengths[:10]
        wl_center = processor.wavelength_norm_params['center']
        wl_scale = processor.wavelength_norm_params['scale']
        wl_norm_test = (original_wl - wl_center) / wl_scale
        wl_recovered = processor.denormalize_wavelength(wl_norm_test)
        
        np.testing.assert_allclose(original_wl, wl_recovered, rtol=1e-6)
    
    def test_get_data_summary(self, processor):
        """Test data summary generation."""
        # Test summary before loading data
        summary_empty = processor.get_data_summary()
        assert summary_empty["status"] == "No data loaded"
        
        # Test summary after full processing
        processor.load_and_validate_data()
        processor.extract_spectral_components()
        processor.compute_beer_lambert_components()
        processor.normalize_inputs()
        
        summary = processor.get_data_summary()
        
        # Check required fields
        assert "data_shape" in summary
        assert "wavelength_range" in summary
        assert "concentrations" in summary
        assert "baseline_stats" in summary
        assert "differential_absorption_stats" in summary
        assert "normalization_params" in summary
        
        # Check normalization parameters structure
        norm_params = summary["normalization_params"]
        assert "wavelength" in norm_params
        assert "concentration" in norm_params
        assert "absorbance" in norm_params
    
    def test_error_handling_invalid_sequence(self):
        """Test error handling when methods called in wrong sequence."""
        processor = UVVisDataProcessor("dummy.csv", validate_data=False)
        
        # Should fail if trying to extract before loading
        with pytest.raises(ValueError, match="Data not loaded"):
            processor.extract_spectral_components()
        
        # Should fail if trying to compute Beer-Lambert before extraction
        with pytest.raises(ValueError, match="Spectral components not extracted"):
            processor.compute_beer_lambert_components()
        
        # Should fail if trying to normalize before computation
        with pytest.raises(ValueError, match="Components not computed"):
            processor.normalize_inputs()


class TestConvenienceFunction:
    """Test suite for convenience function."""
    
    def test_load_uvvis_data_function(self, sample_csv_file):
        """Test the load_uvvis_data convenience function."""
        processor = load_uvvis_data(sample_csv_file, validate=True)
        
        assert isinstance(processor, UVVisDataProcessor)
        assert processor.raw_data is not None
        assert processor.wavelengths is not None
        assert processor.baseline_absorption is not None
        
        # Should be able to create training data immediately
        X_train, y_train = processor.create_training_data()
        assert X_train.shape[1] == 2  # wavelength, concentration
        assert y_train.shape[1] == 1  # differential absorption
    
    def test_load_uvvis_data_no_validation(self, sample_csv_file):
        """Test convenience function without validation."""
        processor = load_uvvis_data(sample_csv_file, validate=False)
        
        assert processor.validate_data is False
        assert processor.raw_data is not None


class TestDataPhysicsValidation:
    """Test physics-related aspects of data processing."""
    
    @pytest.fixture
    def physics_data(self):
        """Create data that follows Beer-Lambert law exactly."""
        wavelengths = np.linspace(200, 800, 101)
        concentrations = [0, 10, 20, 30, 40, 60]
        
        # Define extinction coefficient as function of wavelength
        epsilon = 0.01 + 0.005 * np.exp(-(wavelengths - 400)**2 / 10000)
        baseline = 0.1 * np.ones_like(wavelengths)
        
        data = {'Wavelength': wavelengths}
        for conc in concentrations:
            if conc == 0:
                data[str(conc)] = baseline
            else:
                data[str(conc)] = baseline + epsilon * 1.0 * conc
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            yield f.name, epsilon, baseline
        os.unlink(f.name)
    
    def test_beer_lambert_law_validation(self, physics_data):
        """Test that processed data follows Beer-Lambert law."""
        csv_file, epsilon_true, baseline_true = physics_data
        
        processor = UVVisDataProcessor(csv_file)
        processor.load_and_validate_data()
        processor.extract_spectral_components()
        processor.compute_beer_lambert_components()
        
        # Check baseline extraction
        np.testing.assert_allclose(
            processor.baseline_absorption, 
            baseline_true, 
            rtol=1e-6, 
            err_msg="Baseline not extracted correctly"
        )
        
        # Check Beer-Lambert linearity in differential absorption
        for i, wavelength in enumerate(processor.wavelengths):
            diff_abs_at_wavelength = processor.differential_absorption[i, :]
            concentrations = processor.concentrations_nonzero
            
            # Should be linear: ΔA = ε × b × c
            expected_slope = epsilon_true[i] * 1.0  # b = 1.0 cm
            expected_diff_abs = expected_slope * concentrations
            
            np.testing.assert_allclose(
                diff_abs_at_wavelength,
                expected_diff_abs,
                rtol=1e-6,
                err_msg=f"Beer-Lambert linearity violated at wavelength {wavelength:.1f} nm"
            )
    
    def test_concentration_linearity(self, physics_data):
        """Test that differential absorption is linear with concentration."""
        csv_file, epsilon_true, baseline_true = physics_data
        
        processor = UVVisDataProcessor(csv_file)
        processor.load_and_validate_data()
        processor.extract_spectral_components()
        processor.compute_beer_lambert_components()
        
        # For each wavelength, check linearity
        concentrations = processor.concentrations_nonzero
        
        for i in range(len(processor.wavelengths)):
            diff_abs = processor.differential_absorption[i, :]
            
            # Fit linear regression
            slope = np.polyfit(concentrations, diff_abs, 1)[0]
            
            # Check that the relationship is perfectly linear (R² = 1)
            predicted = slope * concentrations
            r_squared = 1 - np.sum((diff_abs - predicted)**2) / np.sum((diff_abs - np.mean(diff_abs))**2)
            
            assert r_squared > 0.999999, f"Concentration linearity violated at wavelength index {i}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
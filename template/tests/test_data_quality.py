"""
Tests for data quality and validation.

These tests ensure the generated data meets quality standards.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from __project_name__.data_generator import TelecomDataGenerator
from __project_name__.config import RAW_DATA_DIR


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    generator = TelecomDataGenerator(seed=42, n_samples=1000)
    return generator.generate()


class TestDataQuality:
    """Test suite for data quality checks."""
    
    def test_no_missing_values(self, sample_data):
        """Ensure no missing values in critical columns."""
        critical_cols = ["timestamp", "sinr_db", "throughput_mbps", "qoe_mos"]
        for col in critical_cols:
            if col in sample_data.columns:
                assert sample_data[col].isna().sum() == 0, f"Missing values found in {col}"
    
    def test_data_types(self, sample_data):
        """Validate data types."""
        assert pd.api.types.is_datetime64_any_dtype(sample_data["timestamp"])
        assert pd.api.types.is_numeric_dtype(sample_data["sinr_db"])
        assert pd.api.types.is_numeric_dtype(sample_data["throughput_mbps"])
    
    def test_value_ranges(self, sample_data):
        """Ensure values are within realistic ranges."""
        # SINR should be between -5 and 25 dB
        if "sinr_db" in sample_data.columns:
            assert sample_data["sinr_db"].min() >= -5
            assert sample_data["sinr_db"].max() <= 25
        
        # Throughput should be positive
        if "throughput_mbps" in sample_data.columns:
            assert sample_data["throughput_mbps"].min() > 0
        
        # QoE MOS should be between 1 and 5
        if "qoe_mos" in sample_data.columns:
            assert sample_data["qoe_mos"].min() >= 1
            assert sample_data["qoe_mos"].max() <= 5
    
    def test_categorical_values(self, sample_data):
        """Validate categorical column values."""
        if "network_type" in sample_data.columns:
            valid_networks = {"4G", "5G"}
            assert set(sample_data["network_type"].unique()).issubset(valid_networks)
        
        if "device_class" in sample_data.columns:
            valid_devices = {"low", "mid", "high"}
            assert set(sample_data["device_class"].unique()).issubset(valid_devices)
    
    def test_correlation_sanity(self, sample_data):
        """Test that key correlations make sense."""
        # Higher SINR should correlate with higher throughput
        if "sinr_db" in sample_data.columns and "throughput_mbps" in sample_data.columns:
            correlation = sample_data[["sinr_db", "throughput_mbps"]].corr().iloc[0, 1]
            assert correlation > 0, "SINR and throughput should be positively correlated"
        
        # Higher congestion should correlate with higher latency
        if "congestion_level" in sample_data.columns and "latency_ms" in sample_data.columns:
            correlation = sample_data[["congestion_level", "latency_ms"]].corr().iloc[0, 1]
            assert correlation > 0, "Congestion and latency should be positively correlated"
    
    def test_sample_size(self, sample_data):
        """Ensure correct number of samples generated."""
        assert len(sample_data) == 1000


class TestDataGenerator:
    """Test suite for data generator functionality."""
    
    def test_generator_reproducibility(self):
        """Test that generator produces same output with same seed."""
        gen1 = TelecomDataGenerator(seed=42, n_samples=100)
        gen2 = TelecomDataGenerator(seed=42, n_samples=100)
        
        df1 = gen1.generate()
        df2 = gen2.generate()
        
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_sinr_generation(self):
        """Test SINR generation."""
        gen = TelecomDataGenerator(seed=42)
        sinr = gen.generate_sinr(1000)
        
        assert len(sinr) == 1000
        assert sinr.min() >= -5
        assert sinr.max() <= 25
    
    def test_throughput_conversion(self):
        """Test SINR to throughput conversion."""
        gen = TelecomDataGenerator(seed=42)
        sinr = np.array([10, 15, 20])
        network_type = np.array(["4G", "5G", "5G"])
        
        throughput = gen.sinr_to_throughput(sinr, network_type)
        
        # 5G should have higher throughput potential
        assert throughput[2] > 0
        assert len(throughput) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

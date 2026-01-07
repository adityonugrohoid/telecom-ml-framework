"""
Domain-informed synthetic data generator for {PROJECT_NAME}.

This module generates realistic telecom data using domain knowledge rather than
off-the-shelf synthetic data tools. The goal is to create data that reflects
real-world network behavior and challenges.

Key Design Principles:
1. Embed telecom domain physics (signal propagation, congestion, QoE)
2. Introduce realistic noise and imperfections
3. Maintain interpretability over complexity
4. Generate data suitable for the specific ML use case
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path

from .config import DATA_GEN_CONFIG, RAW_DATA_DIR, ensure_directories


class TelecomDataGenerator:
    """
    Base class for generating synthetic telecom data.
    
    This should be subclassed for each specific use case (churn, QoE, etc.)
    with customized generation logic.
    """
    
    def __init__(self, seed: int = 42, n_samples: int = 10_000):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
            n_samples: Number of samples to generate
        """
        self.seed = seed
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        
    def generate(self) -> pd.DataFrame:
        """
        Generate the complete dataset.
        
        This is the main entry point. Override in subclasses.
        
        Returns:
            DataFrame with generated data
        """
        raise NotImplementedError("Subclasses must implement generate()")
    
    # ========================================================================
    # DOMAIN PHYSICS HELPERS
    # ========================================================================
    
    def generate_sinr(
        self, 
        n: int,
        base_sinr_db: float = 10.0,
        noise_std: float = 5.0
    ) -> np.ndarray:
        """
        Generate realistic SINR (Signal-to-Interference-plus-Noise Ratio) values.
        
        Domain knowledge:
        - SINR typically ranges from -5 dB (poor) to 25 dB (excellent)
        - Follows roughly normal distribution around cell-edge value
        - Higher SINR → better throughput and QoE
        
        Args:
            n: Number of samples
            base_sinr_db: Mean SINR in dB
            noise_std: Standard deviation
            
        Returns:
            Array of SINR values in dB
        """
        sinr = self.rng.normal(base_sinr_db, noise_std, n)
        return np.clip(sinr, -5, 25)
    
    def sinr_to_throughput(
        self,
        sinr_db: np.ndarray,
        network_type: np.ndarray,
        noise_factor: float = 0.2
    ) -> np.ndarray:
        """
        Convert SINR to throughput using Shannon-like capacity model.
        
        Domain knowledge:
        - 4G: Max ~100 Mbps, typical 10-50 Mbps
        - 5G: Max ~1 Gbps, typical 50-300 Mbps
        - Relationship is logarithmic: throughput ∝ log2(1 + SNR)
        
        Args:
            sinr_db: SINR values in dB
            network_type: Array of "4G" or "5G"
            noise_factor: Random noise to add (0-1)
            
        Returns:
            Throughput in Mbps
        """
        # Convert dB to linear
        sinr_linear = 10 ** (sinr_db / 10)
        
        # Shannon capacity (simplified)
        capacity_factor = np.log2(1 + sinr_linear)
        
        # Network-specific scaling
        max_throughput = np.where(network_type == "5G", 300, 50)
        throughput = capacity_factor * max_throughput / 5  # Normalize
        
        # Add realistic noise
        noise = self.rng.normal(1, noise_factor, len(throughput))
        throughput = throughput * noise
        
        return np.clip(throughput, 0.1, max_throughput)
    
    def generate_congestion_pattern(
        self,
        timestamps: pd.DatetimeIndex
    ) -> np.ndarray:
        """
        Generate realistic network congestion based on time of day.
        
        Domain knowledge:
        - Peak hours: 9-11 AM, 6-9 PM
        - Low congestion: 12-6 AM
        - Weekend patterns differ from weekdays
        
        Args:
            timestamps: Datetime index
            
        Returns:
            Congestion level (0-1, where 1 is high congestion)
        """
        hour = timestamps.hour
        day_of_week = timestamps.dayofweek
        
        # Base diurnal pattern
        # High at 9AM (0.8), peak at 8PM (1.0), low at 3AM (0.2)
        congestion = 0.5 + 0.3 * np.sin((hour - 6) * np.pi / 12)
        
        # Boost during peak hours
        peak_morning = (hour >= 9) & (hour <= 11)
        peak_evening = (hour >= 18) & (hour <= 21)
        congestion = np.where(peak_morning | peak_evening, congestion * 1.3, congestion)
        
        # Weekend effect (lower congestion)
        is_weekend = day_of_week >= 5
        congestion = np.where(is_weekend, congestion * 0.8, congestion)
        
        # Add noise
        noise = self.rng.normal(0, 0.1, len(congestion))
        congestion = congestion + noise
        
        return np.clip(congestion, 0, 1)
    
    def congestion_to_latency(
        self,
        congestion: np.ndarray,
        base_latency_ms: float = 20
    ) -> np.ndarray:
        """
        Map congestion to latency (RTT).
        
        Domain knowledge:
        - Low congestion: ~10-30 ms
        - High congestion: 50-200 ms (bufferbloat)
        - Non-linear relationship (exponential under heavy load)
        
        Args:
            congestion: Congestion level (0-1)
            base_latency_ms: Baseline latency in ms
            
        Returns:
            Latency in milliseconds
        """
        # Exponential increase under congestion
        latency = base_latency_ms * (1 + 5 * congestion ** 2)
        
        # Add jitter
        jitter = self.rng.normal(0, 5, len(latency))
        latency = latency + jitter
        
        return np.clip(latency, 10, 300)
    
    def compute_qoe_mos(
        self,
        throughput_mbps: np.ndarray,
        latency_ms: np.ndarray,
        packet_loss_pct: np.ndarray,
        app_type: np.ndarray
    ) -> np.ndarray:
        """
        Compute QoE as MOS (Mean Opinion Score) based on network KPIs.
        
        Domain knowledge:
        - MOS ranges from 1 (poor) to 5 (excellent)
        - Different apps have different sensitivities
        - Video: very sensitive to packet loss
        - Gaming: very sensitive to latency
        - Browsing: mostly sensitive to throughput
        
        Args:
            throughput_mbps: Throughput in Mbps
            latency_ms: Latency in ms
            packet_loss_pct: Packet loss percentage
            app_type: Application type
            
        Returns:
            MOS score (1-5)
        """
        # Base MOS from throughput (diminishing returns)
        mos_throughput = 1 + 4 * (1 - np.exp(-throughput_mbps / 10))
        
        # Latency penalty
        latency_penalty = np.clip(latency_ms / 100, 0, 2)
        
        # Packet loss penalty
        loss_penalty = packet_loss_pct / 2
        
        # Combine factors
        mos = mos_throughput - latency_penalty - loss_penalty
        
        # App-specific adjustments
        # Video is more sensitive to packet loss
        video_mask = app_type == "video_streaming"
        mos = np.where(video_mask, mos - packet_loss_pct * 0.5, mos)
        
        # Gaming is more sensitive to latency
        gaming_mask = app_type == "gaming"
        mos = np.where(gaming_mask, mos - latency_penalty * 0.5, mos)
        
        return np.clip(mos, 1, 5)
    
    # ========================================================================
    # DATA SAVING
    # ========================================================================
    
    def save(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Save generated data to parquet file.
        
        Args:
            df: DataFrame to save
            filename: Output filename (without extension)
            
        Returns:
            Path to saved file
        """
        ensure_directories()
        output_path = RAW_DATA_DIR / f"{filename}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"✓ Saved {len(df):,} rows to {output_path}")
        return output_path


# ============================================================================
# EXAMPLE USAGE (TO BE CUSTOMIZED PER USE CASE)
# ============================================================================

class ExampleUseCaseGenerator(TelecomDataGenerator):
    """
    Example generator - customize this for your specific use case.
    """
    
    def generate(self) -> pd.DataFrame:
        """Generate example dataset."""
        # 1. Generate base features
        n = self.n_samples
        
        # Network and device properties
        network_type = self.rng.choice(["4G", "5G"], n, p=[0.6, 0.4])
        device_class = self.rng.choice(["low", "mid", "high"], n, p=[0.2, 0.5, 0.3])
        app_type = self.rng.choice(
            ["video_streaming", "browsing", "gaming", "social"],
            n,
            p=[0.3, 0.4, 0.15, 0.15]
        )
        
        # Timestamps (random distribution over 30 days)
        start_date = pd.Timestamp("2024-01-01")
        timestamps = pd.date_range(start_date, periods=n, freq="min")
        timestamps = self.rng.choice(timestamps, n, replace=True)
        
        # 2. Generate physics-based features
        sinr_db = self.generate_sinr(n)
        throughput_mbps = self.sinr_to_throughput(sinr_db, network_type)
        
        congestion = self.generate_congestion_pattern(pd.DatetimeIndex(timestamps))
        latency_ms = self.congestion_to_latency(congestion)
        
        packet_loss_pct = self.rng.exponential(0.5, n)
        packet_loss_pct = np.clip(packet_loss_pct, 0, 5)
        
        # 3. Compute QoE
        mos = self.compute_qoe_mos(throughput_mbps, latency_ms, packet_loss_pct, app_type)
        
        # 4. Create DataFrame
        df = pd.DataFrame({
            "timestamp": timestamps,
            "network_type": network_type,
            "device_class": device_class,
            "app_type": app_type,
            "sinr_db": sinr_db,
            "throughput_mbps": throughput_mbps,
            "latency_ms": latency_ms,
            "packet_loss_pct": packet_loss_pct,
            "congestion_level": congestion,
            "qoe_mos": mos,
        })
        
        return df


def main():
    """Generate and save synthetic data."""
    generator = ExampleUseCaseGenerator(
        seed=DATA_GEN_CONFIG["random_seed"],
        n_samples=DATA_GEN_CONFIG["n_samples"]
    )
    
    print("Generating synthetic telecom data...")
    df = generator.generate()
    
    print("\nData summary:")
    print(df.describe())
    
    print("\nSample rows:")
    print(df.head())
    
    # Save
    generator.save(df, "synthetic_data")
    print("\n✓ Data generation complete!")


if __name__ == "__main__":
    main()

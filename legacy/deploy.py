#!/usr/bin/env python3
"""
TartanIMU Model Deployment Example
===================================

This script demonstrates how to deploy the TartanIMU model in a production environment
with proper error handling, logging, and performance optimization.
"""

import torch
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, Tuple, Union
from model import TartanIMUModel, load_checkpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TartanIMUPredictor:
    """
    Production-ready predictor for TartanIMU model.
    
    Features:
    - Model caching and warmup
    - Input validation
    - Batch processing
    - Error handling
    - Performance monitoring
    """
    
    def __init__(self, 
                 checkpoint_path: str = 'checkpoint_24.pt',
                 device: str = None,
                 warmup: bool = True):
        """
        Initialize the predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
            warmup: Whether to warmup the model on initialization
        """
        self.device = self._get_device(device)
        self.checkpoint_path = Path(checkpoint_path)
        
        # Load model
        self.model = self._load_model()
        
        # Model warmup
        if warmup:
            self._warmup_model()
    
    def _get_device(self, device: str = None) -> torch.device:
        """Get the appropriate device for inference."""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _load_model(self) -> TartanIMUModel:
        """Load and prepare the model."""
        try:
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
            logger.info(f"Loading model from {self.checkpoint_path}")
            model = TartanIMUModel()
            model = load_checkpoint(model, str(self.checkpoint_path), device=self.device)
            model.to(self.device)
            model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _warmup_model(self):
        """Warmup the model to ensure consistent inference times."""
        logger.info("Warming up model...")
        try:
            # Create dummy input
            warmup_input = torch.randn(1, 6, 13).to(self.device)
            
            # Run a few forward passes
            with torch.no_grad():
                for _ in range(3):
                    _ = self.model(warmup_input)
            
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def _validate_input(self, imu_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Validate and preprocess input data."""
        try:
            # Convert to tensor if needed
            if isinstance(imu_data, np.ndarray):
                imu_data = torch.from_numpy(imu_data).float()
            
            # Ensure correct dtype
            if imu_data.dtype != torch.float32:
                imu_data = imu_data.float()
            
            # Handle different input shapes
            if imu_data.dim() == 2:
                # [seq_len, 6] -> [1, 6, seq_len]
                if imu_data.shape[1] != 6:
                    raise ValueError(f"Expected 6 channels, got {imu_data.shape[1]}")
                imu_data = imu_data.transpose(0, 1).unsqueeze(0)
            elif imu_data.dim() == 3:
                if imu_data.shape[2] == 6:
                    # [batch, seq_len, 6] -> [batch, 6, seq_len]
                    imu_data = imu_data.transpose(1, 2)
                elif imu_data.shape[1] == 6:
                    # [batch, 6, seq_len] - already correct
                    pass
                else:
                    raise ValueError(f"Expected 6 channels in dim 1 or 2, got shape {imu_data.shape}")
            else:
                raise ValueError(f"Invalid input dimensions: {imu_data.dim()}. Expected 2 or 3 dimensions.")
            
            return imu_data.to(self.device)
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise ValueError(f"Invalid input data: {e}")
    
    def predict(self, imu_data: Union[np.ndarray, torch.Tensor]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Run inference on IMU data.
        
        Args:
            imu_data: IMU data array/tensor
            
        Returns:
            Dictionary with predictions for each class
        """
        try:
            # Validate input
            input_tensor = self._validate_input(imu_data)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
            
            return outputs
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def predict_batch(self, imu_batch: list) -> list:
        """
        Run inference on a batch of IMU data.
        
        Args:
            imu_batch: List of IMU data arrays
            
        Returns:
            List of predictions
        """
        try:
            # Find max sequence length
            max_len = max(data.shape[0] if data.ndim == 2 else data.shape[1] for data in imu_batch)
            
            # Pad sequences to same length
            padded_batch = []
            for data in imu_batch:
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data).float()
                
                if data.dim() == 2:
                    seq_len = data.shape[0]
                    if seq_len < max_len:
                        padding = torch.zeros(max_len - seq_len, 6)
                        data = torch.cat([data, padding], dim=0)
                elif data.dim() == 3:
                    seq_len = data.shape[1]
                    if seq_len < max_len:
                        padding = torch.zeros(data.shape[0], max_len - seq_len, 6)
                        data = torch.cat([data, padding], dim=1)
                
                padded_batch.append(data)
            
            # Stack into batch
            batch_tensor = torch.stack(padded_batch, dim=0)
            if batch_tensor.shape[-1] == 6:
                batch_tensor = batch_tensor.transpose(1, 2)  # [batch, 6, seq_len]
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(batch_tensor.to(self.device))
            
            return outputs
            
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            'device': str(self.device),
            'checkpoint_path': str(self.checkpoint_path),
            'model_class': self.model.__class__.__name__,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }


def benchmark_model(predictor: TartanIMUPredictor, 
                   batch_sizes: list = [1, 2, 4, 8],
                   seq_lengths: list = [13, 26, 50],
                   num_runs: int = 100):
    """
    Benchmark model performance across different configurations.
    
    Args:
        predictor: Initialized predictor
        batch_sizes: List of batch sizes to test
        seq_lengths: List of sequence lengths to test
        num_runs: Number of runs per configuration
    """
    print("\n" + "="*60)
    print("MODEL BENCHMARK")
    print("="*60)
    
    device_info = predictor.get_model_info()
    print(f"Device: {device_info['device']}")
    print(f"Total parameters: {device_info['num_parameters']:,}")
    print()
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            # Generate test data
            test_data = np.random.randn(batch_size, 6, seq_len).astype(np.float32)
            
            # Warmup
            _ = predictor.predict(test_data)
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                _ = predictor.predict(test_data)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                times.append(time.time() - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            fps = 1.0 / avg_time
            
            print(f"Batch: {batch_size:2d} | Seq: {seq_len:3d} | "
                  f"Time: {avg_time*1000:.2f}±{std_time*1000:.2f}ms | "
                  f"FPS: {fps:.1f}")


def main():
    """Main deployment example."""
    print("TartanIMU Model Deployment Example")
    print("=" * 50)
    
    # Initialize predictor
    predictor = TartanIMUPredictor(warmup=True)
    
    # Print model information
    info = predictor.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Example 1: Single prediction
    print("\n" + "="*50)
    print("Example 1: Single Prediction")
    print("="*50)
    
    # Generate sample IMU data
    seq_len = 13
    imu_data = np.random.randn(seq_len, 6).astype(np.float32)
    
    # Add some realistic IMU patterns
    t = np.linspace(0, 1, seq_len)
    imu_data[:, 0] = 0.1 * np.sin(2 * np.pi * t)  # acc_x
    imu_data[:, 1] = 0.05 * np.cos(2 * np.pi * t)  # acc_y
    imu_data[:, 3] = 0.2 * np.sin(4 * np.pi * t)  # gyro_x
    
    # Run prediction
    outputs = predictor.predict(imu_data)
    
    # Print results
    for class_name, (out2d, out3d, out_z) in outputs.items():
        print(f"\n{class_name.upper()}:")
        print(f"  2D shape: {out2d.squeeze().cpu().numpy().shape}")
        print(f"  3D shape: {out3d.squeeze().cpu().numpy().shape}")
        print(f"  Z shape:  {out_z.squeeze().cpu().numpy().shape}")
    
    # Example 2: Batch prediction
    print("\n" + "="*50)
    print("Example 2: Batch Prediction")
    print("="*50)
    
    # Create batch of different sequences
    batch_data = []
    for i in range(4):
        seq_len = 10 + i * 5
        imu_seq = np.random.randn(seq_len, 6).astype(np.float32)
        batch_data.append(imu_seq)
    
    # Run batch prediction
    outputs = predictor.predict_batch(batch_data)
    
    print(f"Processed batch of {len(batch_data)} sequences")
    for class_name, (out2d, out3d, out_z) in outputs.items():
        print(f"{class_name}: {out2d.shape}")
    
    # Example 3: Benchmark
    print("\n" + "="*50)
    print("Example 3: Performance Benchmark")
    print("="*50)
    
    benchmark_model(predictor, batch_sizes=[1, 2, 4], seq_lengths=[13, 26], num_runs=50)
    
    # Example 4: Error handling
    print("\n" + "="*50)
    print("Example 4: Error Handling")
    print("="*50)
    
    try:
        # Invalid input shape - wrong number of channels
        invalid_data = np.random.randn(5, 5).astype(np.float32)  # Should be 6 channels, got 5
        predictor.predict(invalid_data)
    except ValueError as e:
        print(f"[OK] Caught expected error: {e}")
    
    try:
        # Invalid input shape - wrong number of channels (3D case)
        invalid_data = np.random.randn(2, 10, 5).astype(np.float32)  # Should be 6 channels, got 5
        predictor.predict(invalid_data)
    except ValueError as e:
        print(f"[OK] Caught expected error: {e}")
    
    try:
        # Invalid dimensions - too many dimensions
        invalid_data = np.random.randn(2, 3, 4, 5).astype(np.float32)  # 4D instead of 2D or 3D
        predictor.predict(invalid_data)
    except ValueError as e:
        print(f"[OK] Caught expected error: {e}")
    
    print("\nAll error handling tests passed!")
    print("\nDeployment example completed successfully!")


if __name__ == "__main__":
    main()
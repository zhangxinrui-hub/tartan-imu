#!/usr/bin/env python3
"""
TartanIMU Model Inference Script
=================================

This script demonstrates how to load and use the TartanIMU model for inference.
The model takes IMU data as input and outputs predictions for different object classes.
"""

import torch
import numpy as np
from model import TartanIMUModel, load_checkpoint


def prepare_input_data(imu_data, device='cpu'):
    """
    Prepare IMU data for model input.
    
    Args:
        imu_data: numpy array of shape [seq_len, 6] or [batch, seq_len, 6]
                 The 6 channels typically represent: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        device: 'cpu' or 'cuda'
    
    Returns:
        torch.Tensor: Prepared input tensor of shape [batch, 6, seq_len]
    """
    if isinstance(imu_data, np.ndarray):
        imu_data = torch.from_numpy(imu_data).float()
    
    # Ensure we have at least 2 dimensions
    if imu_data.dim() == 2:
        imu_data = imu_data.unsqueeze(0)  # Add batch dimension
    
    # Transpose to [batch, 6, seq_len] format expected by the model
    if imu_data.shape[-1] == 6:
        # Input is [batch, seq_len, 6], transpose to [batch, 6, seq_len]
        imu_data = imu_data.transpose(1, 2)
    
    return imu_data.to(device)


def run_inference(model, imu_data, device='cpu'):
    """
    Run inference on IMU data.
    
    Args:
        model: Loaded TartanIMUModel
        imu_data: Input IMU data (numpy array or torch tensor)
        device: 'cpu' or 'cuda'
    
    Returns:
        dict: Predictions for each class
    """
    model.eval()
    
    # Prepare input
    input_tensor = prepare_input_data(imu_data, device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
    
    return outputs


def print_predictions(outputs):
    """
    Print predictions in a readable format.
    
    Args:
        outputs: Model outputs dictionary
    """
    print("\nPredictions:")
    print("=" * 50)
    
    for class_name, (out2d, out3d, out_z) in outputs.items():
        print(f"\n{class_name.upper()}:")
        print(f"  2D Output: {out2d.squeeze().cpu().numpy()}")
        print(f"  3D Output: {out3d.squeeze().cpu().numpy()}")
        print(f"  Z Output:  {out_z.squeeze().cpu().numpy()}")


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = TartanIMUModel()
    checkpoint_path = 'checkpoint_24.pt'
    model = load_checkpoint(model, checkpoint_path, device=device)
    model.to(device)
    print("Model loaded successfully!")
    
    # Example 1: Generate random IMU data
    print("\n" + "=" * 50)
    print("Example 1: Random IMU data")
    print("=" * 50)
    
    seq_len = 13  # Optimal sequence length for this model
    batch_size = 1
    
    # Generate random IMU data [batch, 6, seq_len]
    # Channels: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    random_imu = np.random.randn(batch_size, 6, seq_len).astype(np.float32)
    
    # Run inference
    outputs = run_inference(model, random_imu, device)
    print_predictions(outputs)
    
    # Example 2: Process multiple sequences
    print("\n" + "=" * 50)
    print("Example 2: Multiple sequences")
    print("=" * 50)
    
    batch_size = 3
    seq_len = 20
    
    # Generate multiple sequences
    multiple_imu = np.random.randn(batch_size, 6, seq_len).astype(np.float32)
    
    # Run inference
    outputs = run_inference(model, multiple_imu, device)
    print_predictions(outputs)
    
    # Example 3: Load data from file
    print("\n" + "=" * 50)
    print("Example 3: Different input formats")
    print("=" * 50)
    
    # Format 1: [seq_len, 6]
    imu_data_2d = np.random.randn(15, 6).astype(np.float32)
    outputs = run_inference(model, imu_data_2d, device)
    print_predictions(outputs)
    
    # Format 2: [batch, seq_len, 6]
    imu_data_3d = np.random.randn(2, 15, 6).astype(np.float32)
    outputs = run_inference(model, imu_data_3d, device)
    print_predictions(outputs)


if __name__ == "__main__":
    main()
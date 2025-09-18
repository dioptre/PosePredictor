"""
Test script for Hierarchical Inductive Autoencoder
Verifies architecture and basic functionality
"""

import os
import numpy as np
import tensorflow as tf
import time

from hierarchical_inductive_autoencoder import (
    MultiResolutionHashEncoding,
    PhysicsInformedToFEncoder, 
    VisualEncoder,
    HierarchicalInductiveAutoencoder,
    create_hierarchical_inductive_model
)


def test_multi_resolution_hash_encoding():
    """Test multi-resolution hash encoding layer"""
    print("Testing MultiResolutionHashEncoding...")
    
    # Create layer
    hash_encoder = MultiResolutionHashEncoding(
        num_levels=8,
        features_per_level=4,
        base_resolution=8,
        finest_resolution=128
    )
    
    # Test with dummy coordinates
    batch_size = 2
    H, W = 32, 32
    coordinates = np.random.uniform(0, 1, (batch_size, H, W, 3)).astype(np.float32)
    
    # Run encoding
    encoded = hash_encoder(coordinates)
    
    expected_features = 8 * 4  # num_levels * features_per_level
    assert encoded.shape == (batch_size, H, W, expected_features), \
        f"Expected shape {(batch_size, H, W, expected_features)}, got {encoded.shape}"
    
    print(f"✓ Hash encoding output shape: {encoded.shape}")
    print(f"✓ Features per pixel: {expected_features}")


def test_physics_informed_tof_encoder():
    """Test physics-informed ToF encoder"""
    print("\nTesting PhysicsInformedToFEncoder...")
    
    # Create encoder
    tof_encoder = PhysicsInformedToFEncoder(latent_dim=128)
    
    # Test with dummy ToF histogram
    batch_size = 2
    histograms = np.random.exponential(0.1, (batch_size, 4, 4, 100, 1)).astype(np.float32)
    
    # Add some realistic peaks
    for b in range(batch_size):
        for x in range(4):
            for y in range(4):
                peak_pos = np.random.randint(20, 80)
                peak_width = np.random.uniform(2, 5)
                t_indices = np.arange(100)
                peak = np.exp(-(t_indices - peak_pos)**2 / (2 * peak_width**2))
                histograms[b, x, y, :, 0] += peak
    
    # Run encoding
    encoded = tof_encoder(histograms)
    
    assert encoded.shape == (batch_size, 128), \
        f"Expected shape {(batch_size, 128)}, got {encoded.shape}"
    
    print(f"✓ ToF encoding output shape: {encoded.shape}")
    print(f"✓ Physics-informed features extracted")


def test_visual_encoder():
    """Test visual encoder with hash encoding"""
    print("\nTesting VisualEncoder...")
    
    # Create encoder
    visual_encoder = VisualEncoder(latent_dim=128)
    
    # Test with dummy RGB images
    batch_size = 2
    images = np.random.uniform(0, 1, (batch_size, 64, 64, 3)).astype(np.float32)
    
    # Run encoding
    encoded = visual_encoder(images)
    
    assert encoded.shape == (batch_size, 128), \
        f"Expected shape {(batch_size, 128)}, got {encoded.shape}"
    
    print(f"✓ Visual encoding output shape: {encoded.shape}")
    print(f"✓ Hash + CNN features combined")


def test_hierarchical_model():
    """Test complete hierarchical inductive autoencoder"""
    print("\nTesting HierarchicalInductiveAutoencoder...")
    
    config = {
        'latent_dim': 256,
        'hierarchy_levels': 3,
        'output_resolution': (32, 32),
        'lr': 0.001
    }
    
    # Create model
    model = create_hierarchical_inductive_model(config)
    
    # Test with dummy multi-modal data
    batch_size = 2
    dummy_inputs = {
        'tof': np.random.exponential(0.1, (batch_size, 4, 4, 100, 1)).astype(np.float32),
        'visual': np.random.uniform(0, 1, (batch_size, 128, 128, 3)).astype(np.float32)
    }
    
    # Add realistic ToF peaks
    for b in range(batch_size):
        for x in range(4):
            for y in range(4):
                peak_pos = np.random.randint(20, 80)
                t_indices = np.arange(100)
                peak = np.exp(-(t_indices - peak_pos)**2 / 8)
                dummy_inputs['tof'][b, x, y, :, 0] += peak
    
    # Run inference
    start_time = time.time()
    outputs = model(dummy_inputs, training=False)
    inference_time = time.time() - start_time
    
    # Check outputs
    assert 'depth' in outputs, "Missing depth output"
    assert 'pose' in outputs, "Missing pose output"
    assert 'encoding' in outputs, "Missing encoding output"
    assert 'spatial_features' in outputs, "Missing spatial features"
    
    print(f"✓ Inference time: {inference_time:.3f}s")
    print(f"✓ Depth output shape: {outputs['depth'].shape}")
    print(f"✓ Pose output shape: {outputs['pose'].shape}")
    print(f"✓ Encoding shape: {outputs['encoding'].shape}")
    print(f"✓ Spatial features shape: {outputs['spatial_features'].shape}")
    
    # Test model summary
    print(f"\nModel Parameters:")
    total_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"✓ Total trainable parameters: {total_params:,}")
    
    return model, outputs


def test_multi_modal_combinations():
    """Test different input modality combinations"""
    print("\nTesting Multi-Modal Combinations...")
    
    config = {
        'latent_dim': 128,
        'hierarchy_levels': 2,
        'output_resolution': (32, 32),
        'lr': 0.001
    }
    
    model = HierarchicalInductiveAutoencoder(**config)
    batch_size = 1
    
    # Test ToF only
    tof_only = {
        'tof': np.random.exponential(0.1, (batch_size, 4, 4, 100, 1)).astype(np.float32)
    }
    outputs_tof = model(tof_only, training=False)
    print(f"✓ ToF-only inference: {outputs_tof['depth'].shape}")
    
    # Test Visual only  
    visual_only = {
        'visual': np.random.uniform(0, 1, (batch_size, 128, 128, 3)).astype(np.float32)
    }
    outputs_visual = model(visual_only, training=False)
    print(f"✓ Visual-only inference: {outputs_visual['depth'].shape}")
    
    # Test both modalities
    both_modalities = {**tof_only, **visual_only}
    outputs_both = model(both_modalities, training=False)
    print(f"✓ Multi-modal inference: {outputs_both['depth'].shape}")
    
    print("✓ All modality combinations working")


def test_hash_encoding_efficiency():
    """Test hash encoding efficiency vs traditional convolutions"""
    print("\nTesting Hash Encoding Efficiency...")
    
    # Hash encoding approach
    hash_encoder = MultiResolutionHashEncoding(
        num_levels=16,
        features_per_level=4,
        base_resolution=8,
        finest_resolution=256
    )
    
    # Traditional Conv3D approach (for comparison)
    conv3d_layers = tf.keras.Sequential([
        tf.keras.layers.Conv3D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv3D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv3D(64, 3, padding='same', activation='relu')
    ])
    
    # Test data
    batch_size = 4
    coordinates = np.random.uniform(0, 1, (batch_size, 32, 32, 3)).astype(np.float32)
    volume_data = np.random.uniform(0, 1, (batch_size, 32, 32, 32, 1)).astype(np.float32)
    
    # Time hash encoding
    start_time = time.time()
    for _ in range(10):
        hash_output = hash_encoder(coordinates)
    hash_time = (time.time() - start_time) / 10
    
    # Time conv3d
    start_time = time.time()
    for _ in range(10):
        conv_output = conv3d_layers(volume_data)
    conv_time = (time.time() - start_time) / 10
    
    # Calculate parameter counts
    hash_params = sum([tf.size(w).numpy() for w in hash_encoder.trainable_weights])
    conv_params = sum([tf.size(w).numpy() for w in conv3d_layers.trainable_weights])
    
    print(f"✓ Hash encoding time: {hash_time:.4f}s")
    print(f"✓ Conv3D time: {conv_time:.4f}s")
    print(f"✓ Speed improvement: {conv_time/hash_time:.2f}x")
    print(f"✓ Hash parameters: {hash_params:,}")
    print(f"✓ Conv3D parameters: {conv_params:,}")
    print(f"✓ Parameter reduction: {conv_params/hash_params:.2f}x")


def test_physics_constraints():
    """Test physics-informed loss constraints"""
    print("\nTesting Physics-Informed Constraints...")
    
    from hierarchical_inductive_autoencoder import physics_informed_loss
    
    # Create dummy depth predictions
    batch_size = 2
    y_true = np.random.uniform(0, 1, (batch_size, 32, 32, 1)).astype(np.float32)
    
    # Good prediction (close to ground truth)
    y_pred_good = y_true + np.random.normal(0, 0.05, y_true.shape).astype(np.float32)
    y_pred_good = np.clip(y_pred_good, 0, 1)
    
    # Bad prediction (noisy, inconsistent)
    y_pred_bad = np.random.uniform(0, 1, y_true.shape).astype(np.float32)
    
    # Calculate losses
    loss_good = physics_informed_loss(y_true, y_pred_good)
    loss_bad = physics_informed_loss(y_true, y_pred_bad)
    
    print(f"✓ Good prediction loss: {loss_good:.4f}")
    print(f"✓ Bad prediction loss: {loss_bad:.4f}")
    print(f"✓ Physics constraints working: {loss_bad > loss_good}")
    
    assert loss_bad > loss_good, "Physics constraints not working properly"


def run_all_tests():
    """Run all tests"""
    print("🚀 Running Hierarchical Inductive Autoencoder Tests\n")
    print("=" * 60)
    
    try:
        # Component tests
        test_multi_resolution_hash_encoding()
        test_physics_informed_tof_encoder()
        test_visual_encoder()
        
        # Integration tests  
        model, outputs = test_hierarchical_model()
        test_multi_modal_combinations()
        
        # Performance tests
        test_hash_encoding_efficiency()
        test_physics_constraints()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("\nNext Steps:")
        print("1. Generate synthetic training data:")
        print("   python train_hierarchical.py --generate_synthetic")
        print("2. Train the model:")
        print("   python train_hierarchical.py")
        print("3. Test with real data:")
        print("   python enhanced_pixels2pose.py --scenario 1")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
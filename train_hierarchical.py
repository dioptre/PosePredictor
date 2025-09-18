"""
Training script for Hierarchical Inductive Autoencoder
Supports multi-modal training with ToF + Visual data
"""

import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
import cv2
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

from hierarchical_inductive_autoencoder import create_hierarchical_inductive_model, physics_informed_loss
from enhanced_pixels2pose import EnhancedPixels2Pose


class HierarchicalTrainingDataGenerator(tf.keras.utils.Sequence):
    """
    Data generator for hierarchical model training
    """
    def __init__(self, 
                 data_paths,
                 batch_size=4,
                 visual_size=(128, 128),
                 augment=True,
                 shuffle=True):
        self.data_paths = data_paths
        self.batch_size = batch_size
        self.visual_size = visual_size
        self.augment = augment
        self.shuffle = shuffle
        
        self.indexes = np.arange(len(self.data_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return len(self.data_paths) // self.batch_size
    
    def __getitem__(self, index):
        # Generate batch indices
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.data_paths[k] for k in batch_indexes]
        
        # Generate batch data
        return self._generate_batch(batch_paths)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def _generate_batch(self, batch_paths):
        """Generate a batch of training data"""
        batch_tof = []
        batch_visual = []
        batch_depth_gt = []
        batch_pose_gt = []
        
        for data_path in batch_paths:
            # Load data
            data = sio.loadmat(data_path)
            histogram = data['histogram']  # ToF histogram
            rgb_image = data['reference_RGB']  # Reference image
            
            # For ground truth, we'll use the reference depth and generate pose targets
            # In practice, you'd have proper ground truth annotations
            depth_gt = self._generate_depth_ground_truth(histogram)
            pose_gt = self._generate_pose_ground_truth(rgb_image)
            
            # Preprocess ToF data
            tof_processed = self._preprocess_tof(histogram)
            
            # Preprocess visual data
            visual_processed = self._preprocess_visual(rgb_image)
            
            # Data augmentation
            if self.augment:
                tof_processed, visual_processed, depth_gt, pose_gt = self._augment_data(
                    tof_processed, visual_processed, depth_gt, pose_gt
                )
            
            batch_tof.append(tof_processed)
            batch_visual.append(visual_processed)
            batch_depth_gt.append(depth_gt)
            batch_pose_gt.append(pose_gt)
        
        # Convert to numpy arrays
        batch_tof = np.array(batch_tof)
        batch_visual = np.array(batch_visual)
        batch_depth_gt = np.array(batch_depth_gt)
        batch_pose_gt = np.array(batch_pose_gt)
        
        # Prepare inputs and outputs
        inputs = {
            'tof': batch_tof,
            'visual': batch_visual
        }
        
        outputs = {
            'depth': batch_depth_gt,
            'pose': batch_pose_gt
        }
        
        return inputs, outputs
    
    def _preprocess_tof(self, histogram):
        """Preprocess ToF histogram data"""
        # Ensure correct shape and normalize
        if len(histogram.shape) == 3:
            histogram = np.expand_dims(histogram, axis=-1)
        
        histogram = histogram.astype(np.float32)
        histogram = (histogram - histogram.min()) / (histogram.max() - histogram.min() + 1e-8)
        
        return histogram[0]  # Remove batch dimension added by original data
    
    def _preprocess_visual(self, rgb_image):
        """Preprocess visual data"""
        # Resize and normalize
        rgb_resized = cv2.resize(rgb_image, self.visual_size)
        rgb_normalized = rgb_resized.astype(np.float32) / 255.0
        
        return rgb_normalized
    
    def _generate_depth_ground_truth(self, histogram):
        """Generate depth ground truth from histogram (placeholder)"""
        # This is a placeholder - in practice you'd have proper ground truth
        # For now, we'll create a simple depth map from the histogram peaks
        
        # Find peak positions in temporal dimension
        peak_positions = np.argmax(histogram[0], axis=-1)  # (4, 4)
        
        # Convert to normalized depth values
        depth_map = peak_positions.astype(np.float32) / 100.0  # Normalize to 0-1
        
        # Resize to target resolution (32, 32)
        depth_resized = cv2.resize(depth_map, (32, 32))
        depth_resized = np.expand_dims(depth_resized, axis=-1)
        
        return depth_resized
    
    def _generate_pose_ground_truth(self, rgb_image):
        """Generate pose ground truth (placeholder)"""
        # This is a placeholder - in practice you'd use OpenPose or manual annotations
        # For now, create dummy heatmaps
        
        pose_heatmaps = np.zeros((32, 32, 18), dtype=np.float32)
        
        # Add some dummy pose points (this is just for demonstration)
        # In real training, you'd use proper pose annotations
        center_x, center_y = 16, 16
        for joint_idx in range(18):
            # Create Gaussian heatmap for each joint
            y, x = np.ogrid[:32, :32]
            joint_x = center_x + np.random.randint(-5, 5)
            joint_y = center_y + np.random.randint(-5, 5)
            
            # Ensure within bounds
            joint_x = np.clip(joint_x, 0, 31)
            joint_y = np.clip(joint_y, 0, 31)
            
            heatmap = np.exp(-((x - joint_x)**2 + (y - joint_y)**2) / (2 * 2**2))
            pose_heatmaps[:, :, joint_idx] = heatmap
        
        return pose_heatmaps
    
    def _augment_data(self, tof_data, visual_data, depth_gt, pose_gt):
        """Apply data augmentation"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            tof_data = np.flip(tof_data, axis=1)
            visual_data = np.flip(visual_data, axis=1)
            depth_gt = np.flip(depth_gt, axis=1)
            pose_gt = np.flip(pose_gt, axis=1)
        
        # Random noise addition to ToF data
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.01, tof_data.shape)
            tof_data = np.clip(tof_data + noise, 0, 1)
        
        # Random brightness adjustment for visual data
        if np.random.random() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            visual_data = np.clip(visual_data * brightness_factor, 0, 1)
        
        return tof_data, visual_data, depth_gt, pose_gt


def create_training_config():
    """Create default training configuration"""
    return {
        'model': {
            'latent_dim': 512,
            'hierarchy_levels': 4,
            'output_resolution': (32, 32),
            'use_visual_fusion': True,
            'use_physics_constraints': True
        },
        'training': {
            'batch_size': 4,
            'epochs': 100,
            'learning_rate': 0.0005,
            'validation_split': 0.2,
            'visual_size': (128, 128),
            'augment': True
        },
        'paths': {
            'data_dir': 'training_data',
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs'
        }
    }


def collect_training_data(data_directories):
    """Collect all .mat files for training"""
    data_files = []
    
    for data_dir in data_directories:
        data_path = Path(data_dir)
        if data_path.exists():
            mat_files = list(data_path.glob('**/data.mat'))
            data_files.extend([str(f) for f in mat_files])
            print(f"Found {len(mat_files)} data files in {data_dir}")
        else:
            print(f"Warning: Directory {data_dir} does not exist")
    
    print(f"Total training files: {len(data_files)}")
    return data_files


def train_hierarchical_model(config):
    """Train the hierarchical inductive autoencoder"""
    
    # Create directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # Collect training data
    data_directories = ['1_PEOPLE', '2_PEOPLE', '3_PEOPLE']
    training_files = collect_training_data(data_directories)
    
    if len(training_files) < 10:
        print("Warning: Very few training files found. Consider generating synthetic data.")
    
    # Split into train/validation
    np.random.shuffle(training_files)
    split_idx = int(len(training_files) * (1 - config['training']['validation_split']))
    train_files = training_files[:split_idx]
    val_files = training_files[split_idx:]
    
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    
    # Create data generators
    train_generator = HierarchicalTrainingDataGenerator(
        train_files,
        batch_size=config['training']['batch_size'],
        visual_size=config['training']['visual_size'],
        augment=config['training']['augment'],
        shuffle=True
    )
    
    val_generator = HierarchicalTrainingDataGenerator(
        val_files,
        batch_size=config['training']['batch_size'],
        visual_size=config['training']['visual_size'],
        augment=False,
        shuffle=False
    )
    
    # Create model
    model = create_hierarchical_inductive_model(config['model'])
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(config['paths']['checkpoint_dir'], 'hierarchical_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=config['paths']['log_dir'],
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config['training']['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(config['paths']['checkpoint_dir'], 'hierarchical_final.h5')
    model.save_weights(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Save training history
    history_path = os.path.join(config['paths']['log_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    
    # Plot training curves
    plot_training_history(history, config['paths']['log_dir'])
    
    return model, history


def plot_training_history(history, log_dir):
    """Plot and save training history"""
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Depth accuracy plot
    if 'depth_mae' in history.history:
        plt.subplot(1, 3, 2)
        plt.plot(history.history['depth_mae'], label='Training Depth MAE')
        plt.plot(history.history['val_depth_mae'], label='Validation Depth MAE')
        plt.title('Depth Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
    
    # Pose accuracy plot
    if 'pose_mae' in history.history:
        plt.subplot(1, 3, 3)
        plt.plot(history.history['pose_mae'], label='Training Pose MAE')
        plt.plot(history.history['val_pose_mae'], label='Validation Pose MAE')
        plt.title('Pose Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_curves.png'), dpi=150)
    plt.show()


def generate_synthetic_training_data(num_samples=1000, output_dir='synthetic_data'):
    """Generate synthetic training data for initial training"""
    print(f"Generating {num_samples} synthetic training samples...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Generate synthetic ToF histogram
        histogram = np.random.exponential(0.1, (1, 4, 4, 100))
        
        # Add some realistic peaks
        for x in range(4):
            for y in range(4):
                # Add 1-2 peaks per pixel
                num_peaks = np.random.randint(1, 3)
                for peak in range(num_peaks):
                    peak_pos = np.random.randint(10, 90)
                    peak_width = np.random.uniform(2, 8)
                    peak_amp = np.random.uniform(0.1, 1.0)
                    
                    # Add Gaussian peak
                    t_indices = np.arange(100)
                    gaussian_peak = peak_amp * np.exp(
                        -(t_indices - peak_pos)**2 / (2 * peak_width**2)
                    )
                    histogram[0, x, y, :] += gaussian_peak
        
        # Generate synthetic RGB image
        rgb_image = np.random.uniform(0, 255, (224, 200, 3)).astype(np.uint8)
        
        # Add some structure to the image
        center_x, center_y = 112, 100
        for _ in range(np.random.randint(1, 4)):  # 1-3 people
            person_x = center_x + np.random.randint(-50, 50)
            person_y = center_y + np.random.randint(-30, 30)
            
            # Draw simple ellipse for person
            cv2.ellipse(rgb_image, (person_y, person_x), (20, 40), 0, 0, 360, 
                       (np.random.randint(100, 255), np.random.randint(100, 255), 
                        np.random.randint(100, 255)), -1)
        
        # Save synthetic data
        synthetic_data = {
            'histogram': histogram,
            'reference_RGB': rgb_image
        }
        
        output_path = os.path.join(output_dir, f'synthetic_{i:04d}.mat')
        sio.savemat(output_path, synthetic_data)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_samples} samples")
    
    print(f"Synthetic data saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Hierarchical Inductive Autoencoder')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--generate_synthetic', action='store_true',
                       help='Generate synthetic training data')
    parser.add_argument('--synthetic_samples', type=int, default=1000,
                       help='Number of synthetic samples to generate')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from: {args.config}")
    else:
        config = create_training_config()
        print("Using default configuration")
    
    # Generate synthetic data if requested
    if args.generate_synthetic:
        generate_synthetic_training_data(
            num_samples=args.synthetic_samples,
            output_dir='synthetic_training_data'
        )
        return
    
    # Train model
    model, history = train_hierarchical_model(config)
    
    print("\nTraining completed!")
    print("Next steps:")
    print("1. Check training curves in the logs directory")
    print("2. Test the model with: python enhanced_pixels2pose.py --scenario 1 --weights_path checkpoints/hierarchical_best.h5")
    print("3. Fine-tune on real data if using synthetic pre-training")


if __name__ == "__main__":
    main()
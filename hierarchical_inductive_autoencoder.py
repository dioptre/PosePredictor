"""
Hierarchical Inductive Autoencoder with Multi-Resolution Hash Encoding
for ToF, Visual, and Multi-Modal Sensor Fusion

Combines:
- Multi-resolution hash encoding (NeRF-style) for efficient feature representation
- Hierarchical autoencoder for multi-scale processing
- Physics-informed inductive biases for ToF sensors
- Modular fusion for visual, ToF, IMU, and other sensor modalities
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
import numpy as np
import math


class MultiResolutionHashEncoding(tf.keras.layers.Layer):
    """
    Multi-resolution hash encoding inspired by InstantNGP/NeRF
    Efficiently encode spatial-temporal coordinates with hash tables
    """
    def __init__(self, 
                 num_levels=16,
                 features_per_level=2,
                 log2_hashmap_size=19,
                 base_resolution=16,
                 finest_resolution=2048,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        
        # Growth factor between resolution levels
        self.growth_factor = np.exp(
            (np.log(finest_resolution) - np.log(base_resolution)) / (num_levels - 1)
        )
        
        # Hash tables for each level
        self.hash_tables = []
        for level in range(num_levels):
            resolution = int(np.floor(base_resolution * (self.growth_factor ** level)))
            hashmap_size = min(resolution ** 3, 2 ** log2_hashmap_size)  # 3D grid
            
            hash_table = self.add_weight(
                name=f'hash_table_level_{level}',
                shape=(hashmap_size, features_per_level),
                initializer='uniform',
                trainable=True
            )
            self.hash_tables.append(hash_table)
    
    def hash_function(self, coordinates, level):
        """3D spatial hash function for coordinates"""
        resolution = int(np.floor(self.base_resolution * (self.growth_factor ** level)))
        
        # Scale coordinates to resolution
        scaled_coords = coordinates * resolution
        
        # Floor to get grid coordinates
        grid_coords = tf.cast(tf.floor(scaled_coords), tf.int32)
        
        # 3D hash: combine x, y, t coordinates
        x, y, t = tf.split(grid_coords, 3, axis=-1)
        
        # Simple hash function (can be improved)
        hash_val = (x * 73856093) ^ (y * 19349663) ^ (t * 83492791)
        hash_val = tf.abs(hash_val) % self.hash_tables[level].shape[0]
        
        return tf.squeeze(hash_val, axis=-1)
    
    def call(self, coordinates):
        """
        coordinates: (batch, ..., 3) - normalized (x, y, t) coordinates
        Returns: (batch, ..., num_levels * features_per_level)
        """
        batch_shape = tf.shape(coordinates)[:-1]
        flat_coords = tf.reshape(coordinates, (-1, 3))
        
        encoded_features = []
        
        for level in range(self.num_levels):
            # Get hash indices
            hash_indices = self.hash_function(flat_coords, level)
            
            # Lookup features from hash table
            level_features = tf.gather(self.hash_tables[level], hash_indices)
            encoded_features.append(level_features)
        
        # Concatenate all levels
        all_features = tf.concat(encoded_features, axis=-1)
        
        # Reshape back to original batch shape
        output_shape = tf.concat([batch_shape, [self.num_levels * self.features_per_level]], axis=0)
        return tf.reshape(all_features, output_shape)


class PhysicsInformedToFEncoder(tf.keras.layers.Layer):
    """
    Physics-informed encoder for ToF histogram data
    Incorporates knowledge about light propagation, multi-path returns, etc.
    """
    def __init__(self, latent_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        
        # Gaussian peak decomposition
        self.gaussian_analyzer = Dense(64, activation='relu', name='gaussian_analyzer')
        self.peak_detector = Dense(32, activation='relu', name='peak_detector')
        
        # Multi-path separation
        self.multipath_separator = Conv3D(16, (3,3,5), activation='relu', name='multipath_sep')
        self.temporal_analyzer = Conv1D(32, 5, activation='relu', name='temporal_analyzer')
        
        # Physics constraints
        self.physics_projector = Dense(latent_dim, activation='tanh', name='physics_proj')
        
    def gaussian_peak_analysis(self, histograms):
        """Extract Gaussian peak parameters (amplitude, position, width)"""
        # histograms: (batch, 4, 4, 100, 1)
        batch_size = tf.shape(histograms)[0]
        
        # Flatten spatial dimensions for analysis
        flat_histograms = tf.reshape(histograms, (batch_size, 16, 100))
        
        # Analyze each pixel's temporal histogram
        gaussian_features = self.gaussian_analyzer(flat_histograms)
        peak_params = self.peak_detector(gaussian_features)
        
        return peak_params
    
    def multipath_analysis(self, histograms):
        """Separate direct vs indirect light returns"""
        # Apply 3D convolution to capture spatial-temporal patterns
        multipath_features = self.multipath_separator(histograms)
        
        # Analyze temporal patterns
        reshaped = tf.reshape(multipath_features, 
                            (tf.shape(multipath_features)[0], -1, tf.shape(multipath_features)[-2]))
        temporal_features = self.temporal_analyzer(reshaped)
        
        return temporal_features
    
    def call(self, histograms):
        """
        histograms: (batch, 4, 4, 100, 1) - ToF temporal histograms
        Returns: (batch, latent_dim) - physics-informed encoding
        """
        # Gaussian peak analysis
        gaussian_features = self.gaussian_peak_analysis(histograms)
        
        # Multi-path analysis  
        multipath_features = self.multipath_analysis(histograms)
        
        # Combine features
        combined_features = tf.concat([
            tf.reshape(gaussian_features, (tf.shape(gaussian_features)[0], -1)),
            tf.reshape(multipath_features, (tf.shape(multipath_features)[0], -1))
        ], axis=-1)
        
        # Project to physics-constrained latent space
        physics_encoding = self.physics_projector(combined_features)
        
        return physics_encoding


class VisualEncoder(tf.keras.layers.Layer):
    """Visual encoder for RGB/depth images using hash encoding"""
    def __init__(self, latent_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        
        # Hash encoding for spatial features
        self.spatial_hash_encoder = MultiResolutionHashEncoding(
            num_levels=12,
            features_per_level=4,
            base_resolution=8,
            finest_resolution=256
        )
        
        # CNN backbone
        self.conv_layers = [
            Conv2D(32, 3, activation='relu', name='vis_conv1'),
            MaxPooling2D(2),
            Conv2D(64, 3, activation='relu', name='vis_conv2'), 
            MaxPooling2D(2),
            Conv2D(128, 3, activation='relu', name='vis_conv3'),
            GlobalAveragePooling2D()
        ]
        
        self.visual_projector = Dense(latent_dim, activation='tanh', name='visual_proj')
    
    def call(self, images):
        """
        images: (batch, H, W, C) - RGB or depth images
        Returns: (batch, latent_dim) - visual encoding
        """
        # Generate coordinate grid for hash encoding
        batch_size = tf.shape(images)[0]
        H, W = tf.shape(images)[1], tf.shape(images)[2]
        
        # Create normalized coordinate grid
        y_coords = tf.linspace(0.0, 1.0, H)
        x_coords = tf.linspace(0.0, 1.0, W) 
        yy, xx = tf.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Add temporal dimension (set to 0.5 for static images)
        tt = tf.ones_like(xx) * 0.5
        coordinates = tf.stack([xx, yy, tt], axis=-1)
        coordinates = tf.expand_dims(coordinates, 0)
        coordinates = tf.tile(coordinates, [batch_size, 1, 1, 1])
        
        # Hash encoding of spatial coordinates
        hash_features = self.spatial_hash_encoder(coordinates)
        
        # CNN features
        cnn_features = images
        for layer in self.conv_layers:
            cnn_features = layer(cnn_features)
        
        # Combine hash and CNN features
        # Average pool hash features to match CNN spatial dimensions
        pooled_hash = tf.reduce_mean(tf.reduce_mean(hash_features, axis=1), axis=1)
        
        combined_features = tf.concat([cnn_features, pooled_hash], axis=-1)
        visual_encoding = self.visual_projector(combined_features)
        
        return visual_encoding


class HierarchicalInductiveAutoencoder(tf.keras.Model):
    """
    Main hierarchical autoencoder with multi-modal fusion
    """
    def __init__(self, 
                 latent_dim=256,
                 num_hierarchy_levels=3,
                 output_resolution=(32, 32),
                 **kwargs):
        super().__init__(**kwargs)
        
        self.latent_dim = latent_dim
        self.num_hierarchy_levels = num_hierarchy_levels
        self.output_resolution = output_resolution
        
        # Multi-modal encoders
        self.tof_encoder = PhysicsInformedToFEncoder(latent_dim // 2)
        self.visual_encoder = VisualEncoder(latent_dim // 2)
        
        # Hierarchical processing levels
        self.hierarchy_levels = []
        for level in range(num_hierarchy_levels):
            level_dim = latent_dim // (2 ** level)
            level_processor = Dense(level_dim, activation='relu', 
                                  name=f'hierarchy_level_{level}')
            self.hierarchy_levels.append(level_processor)
        
        # Multi-resolution hash decoder
        self.spatial_hash_decoder = MultiResolutionHashEncoding(
            num_levels=8,
            features_per_level=8,
            base_resolution=4,
            finest_resolution=output_resolution[0]
        )
        
        # Output decoders for different tasks
        self.depth_decoder = self._build_depth_decoder()
        self.pose_decoder = self._build_pose_decoder()
        
        # Fusion layers
        self.modal_fusion = Dense(latent_dim, activation='relu', name='modal_fusion')
        self.hierarchy_fusion = Dense(latent_dim, activation='relu', name='hierarchy_fusion')
    
    def _build_depth_decoder(self):
        """Build decoder for depth estimation"""
        return tf.keras.Sequential([
            Dense(512, activation='relu'),
            Dense(1024, activation='relu'),
            Reshape((32, 32, 1)),
            Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
            Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu'),
            Conv2D(1, 3, padding='same', activation='sigmoid')
        ], name='depth_decoder')
    
    def _build_pose_decoder(self):
        """Build decoder for pose estimation"""  
        return tf.keras.Sequential([
            Dense(512, activation='relu'),
            Dense(18 * 32 * 32, activation='relu'),  # 18 joints
            Reshape((32, 32, 18)),
        ], name='pose_decoder')
    
    def call(self, inputs, training=None):
        """
        inputs: dict with keys 'tof', 'visual', etc.
        """
        encodings = []
        
        # Multi-modal encoding
        if 'tof' in inputs:
            tof_encoding = self.tof_encoder(inputs['tof'])
            encodings.append(tof_encoding)
        
        if 'visual' in inputs:
            visual_encoding = self.visual_encoder(inputs['visual'])
            encodings.append(visual_encoding)
        
        # Fuse modalities
        if len(encodings) > 1:
            fused_encoding = self.modal_fusion(tf.concat(encodings, axis=-1))
        else:
            fused_encoding = encodings[0]
        
        # Hierarchical processing
        hierarchy_features = []
        current_features = fused_encoding
        
        for level_processor in self.hierarchy_levels:
            current_features = level_processor(current_features)
            hierarchy_features.append(current_features)
        
        # Fuse hierarchical features
        final_encoding = self.hierarchy_fusion(tf.concat(hierarchy_features, axis=-1))
        
        # Generate coordinate grid for hash decoding
        batch_size = tf.shape(final_encoding)[0]
        H, W = self.output_resolution
        
        y_coords = tf.linspace(0.0, 1.0, H)
        x_coords = tf.linspace(0.0, 1.0, W)
        yy, xx = tf.meshgrid(y_coords, x_coords, indexing='ij')
        tt = tf.ones_like(xx) * 0.5
        coordinates = tf.stack([xx, yy, tt], axis=-1)
        coordinates = tf.expand_dims(coordinates, 0)
        coordinates = tf.tile(coordinates, [batch_size, 1, 1, 1])
        
        # Hash-based spatial decoding
        hash_features = self.spatial_hash_decoder(coordinates)
        
        # Combine with global encoding
        # Broadcast global encoding to spatial dimensions
        global_spatial = tf.expand_dims(tf.expand_dims(final_encoding, 1), 1)
        global_spatial = tf.tile(global_spatial, [1, H, W, 1])
        
        combined_spatial = tf.concat([hash_features, global_spatial], axis=-1)
        
        # Task-specific outputs
        depth_output = self.depth_decoder(tf.reduce_mean(combined_spatial, axis=[1,2]))
        pose_output = self.pose_decoder(tf.reduce_mean(combined_spatial, axis=[1,2]))
        
        return {
            'depth': depth_output,
            'pose': pose_output,
            'encoding': final_encoding,
            'spatial_features': combined_spatial
        }


def physics_informed_loss(y_true, y_pred, alpha=0.1, beta=0.05):
    """
    Physics-informed loss function for ToF data
    """
    # Standard reconstruction loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Physics constraint 1: Depth smoothness (total variation)
    def total_variation_loss(images):
        pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
        pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]
        return tf.reduce_mean(tf.square(pixel_dif1)) + tf.reduce_mean(tf.square(pixel_dif2))
    
    smoothness_loss = total_variation_loss(y_pred)
    
    # Physics constraint 2: Range consistency
    # ToF sensors have limited range - enforce realistic depth values
    range_loss = tf.reduce_mean(tf.nn.relu(y_pred - 1.0))  # Penalize > max range
    
    # Physics constraint 3: Multi-path consistency
    # Adjacent pixels should have consistent depth relationships
    neighbor_consistency = tf.reduce_mean(tf.abs(
        y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
    ))
    
    total_loss = mse_loss + alpha * smoothness_loss + beta * (range_loss + neighbor_consistency)
    
    return total_loss


def create_hierarchical_inductive_model(config):
    """
    Factory function to create the complete model
    """
    model = HierarchicalInductiveAutoencoder(
        latent_dim=config.get('latent_dim', 256),
        num_hierarchy_levels=config.get('hierarchy_levels', 3),
        output_resolution=config.get('output_resolution', (32, 32))
    )
    
    # Compile with physics-informed loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.get('lr', 0.001)),
        loss={
            'depth': physics_informed_loss,
            'pose': 'mse'
        },
        loss_weights={
            'depth': 1.0,
            'pose': 0.5
        },
        metrics={
            'depth': ['mae'],
            'pose': ['mae']
        }
    )
    
    return model


if __name__ == "__main__":
    # Example usage
    config = {
        'latent_dim': 256,
        'hierarchy_levels': 3,
        'output_resolution': (32, 32),
        'lr': 0.001
    }
    
    model = create_hierarchical_inductive_model(config)
    
    # Test with dummy data
    dummy_inputs = {
        'tof': tf.random.normal((2, 4, 4, 100, 1)),
        'visual': tf.random.normal((2, 64, 64, 3))
    }
    
    outputs = model(dummy_inputs)
    print("Model created successfully!")
    print(f"Depth output shape: {outputs['depth'].shape}")
    print(f"Pose output shape: {outputs['pose'].shape}")
    print(f"Encoding shape: {outputs['encoding'].shape}")
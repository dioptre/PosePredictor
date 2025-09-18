"""
Temporal Prediction Extensions for Hierarchical Inductive Autoencoder
Predicts future poses and motion from current and past frames
"""

import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np

from hierarchical_inductive_autoencoder import (
    MultiResolutionHashEncoding,
    PhysicsInformedToFEncoder,
    VisualEncoder
)


class TemporalHashEncoding(tf.keras.layers.Layer):
    """
    4D hash encoding for spatial-temporal data (x, y, z, t)
    Handles sequence of frames with temporal relationships
    """
    def __init__(self, 
                 num_levels=16,
                 features_per_level=4,
                 log2_hashmap_size=20,
                 base_resolution=8,
                 finest_resolution=512,
                 temporal_resolution=32,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.temporal_resolution = temporal_resolution
        
        # Growth factors for spatial and temporal dimensions
        self.spatial_growth = np.exp(
            (np.log(finest_resolution) - np.log(base_resolution)) / (num_levels - 1)
        )
        
        # 4D hash tables (x, y, z, t)
        self.hash_tables = []
        for level in range(num_levels):
            spatial_res = int(np.floor(base_resolution * (self.spatial_growth ** level)))
            temporal_res = min(temporal_resolution, spatial_res)
            hashmap_size = min(spatial_res ** 3 * temporal_res, 2 ** log2_hashmap_size)
            
            hash_table = self.add_weight(
                name=f'temporal_hash_table_level_{level}',
                shape=(hashmap_size, features_per_level),
                initializer='uniform',
                trainable=True
            )
            self.hash_tables.append(hash_table)
    
    def hash_function_4d(self, coordinates, level):
        """4D spatial-temporal hash function"""
        spatial_res = int(np.floor(self.base_resolution * (self.spatial_growth ** level)))
        temporal_res = min(self.temporal_resolution, spatial_res)
        
        # Scale coordinates
        scaled_coords = coordinates * tf.constant([spatial_res, spatial_res, spatial_res, temporal_res], dtype=tf.float32)
        grid_coords = tf.cast(tf.floor(scaled_coords), tf.int32)
        
        # 4D hash
        x, y, z, t = tf.split(grid_coords, 4, axis=-1)
        hash_val = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791) ^ (t * 50331653)
        hash_val = tf.abs(hash_val) % self.hash_tables[level].shape[0]
        
        return tf.squeeze(hash_val, axis=-1)
    
    def call(self, coordinates):
        """
        coordinates: (batch, seq_len, H, W, 4) - normalized (x, y, z, t)
        Returns: (batch, seq_len, H, W, num_levels * features_per_level)
        """
        original_shape = tf.shape(coordinates)
        batch_size, seq_len = original_shape[0], original_shape[1]
        H, W = original_shape[2], original_shape[3]
        
        # Flatten for processing
        flat_coords = tf.reshape(coordinates, (-1, 4))
        
        encoded_features = []
        for level in range(self.num_levels):
            hash_indices = self.hash_function_4d(flat_coords, level)
            level_features = tf.gather(self.hash_tables[level], hash_indices)
            encoded_features.append(level_features)
        
        # Concatenate and reshape back
        all_features = tf.concat(encoded_features, axis=-1)
        output_features = self.num_levels * self.features_per_level
        
        return tf.reshape(all_features, (batch_size, seq_len, H, W, output_features))


class MotionDynamicsEncoder(tf.keras.layers.Layer):
    """
    Physics-informed encoder for human motion dynamics
    Incorporates biomechanical constraints and motion priors
    """
    def __init__(self, latent_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        
        # Biomechanical constraints
        self.joint_constraint_layer = Dense(64, activation='relu', name='joint_constraints')
        self.velocity_analyzer = Dense(32, activation='relu', name='velocity_analyzer')
        self.acceleration_analyzer = Dense(32, activation='relu', name='acceleration_analyzer')
        
        # Motion pattern recognition
        self.motion_pattern_conv = Conv1D(64, 5, activation='relu', name='motion_patterns')
        self.temporal_attention = MultiHeadAttention(num_heads=4, key_dim=32, name='temporal_attention')
        
        # Physics projector
        self.motion_projector = Dense(latent_dim, activation='tanh', name='motion_proj')
    
    def compute_motion_features(self, pose_sequence):
        """Compute velocity, acceleration, and biomechanical features"""
        # pose_sequence: (batch, seq_len, num_joints, 3)
        
        # Compute velocities (finite differences)
        velocities = pose_sequence[:, 1:] - pose_sequence[:, :-1]
        
        # Compute accelerations
        accelerations = velocities[:, 1:] - velocities[:, :-1]
        
        # Pad to maintain sequence length
        zero_pad = tf.zeros_like(pose_sequence[:, :1])
        velocities = tf.concat([zero_pad, velocities], axis=1)
        
        zero_pad_acc = tf.zeros_like(pose_sequence[:, :2])
        accelerations = tf.concat([zero_pad_acc, accelerations], axis=1)
        
        return velocities, accelerations
    
    def call(self, pose_sequence):
        """
        pose_sequence: (batch, seq_len, num_joints, 3)
        Returns: (batch, latent_dim) - motion dynamics encoding
        """
        batch_size, seq_len, num_joints, _ = tf.shape(pose_sequence)[0], tf.shape(pose_sequence)[1], tf.shape(pose_sequence)[2], tf.shape(pose_sequence)[3]
        
        # Compute motion derivatives
        velocities, accelerations = self.compute_motion_features(pose_sequence)
        
        # Flatten spatial dimensions for temporal processing
        poses_flat = tf.reshape(pose_sequence, (batch_size, seq_len, -1))
        velocities_flat = tf.reshape(velocities, (batch_size, seq_len, -1))
        accelerations_flat = tf.reshape(accelerations, (batch_size, seq_len, -1))
        
        # Analyze biomechanical constraints
        joint_features = self.joint_constraint_layer(poses_flat)
        velocity_features = self.velocity_analyzer(velocities_flat)
        acceleration_features = self.acceleration_analyzer(accelerations_flat)
        
        # Temporal pattern recognition
        combined_features = tf.concat([joint_features, velocity_features, acceleration_features], axis=-1)
        motion_patterns = self.motion_pattern_conv(combined_features)
        
        # Temporal attention for important motion phases
        attended_features, attention_weights = self.temporal_attention(
            motion_patterns, motion_patterns, return_attention_scores=True
        )
        
        # Global motion encoding
        motion_encoding = tf.reduce_mean(attended_features, axis=1)  # Average over time
        final_encoding = self.motion_projector(motion_encoding)
        
        return final_encoding, attention_weights


class TemporalPredictionHead(tf.keras.layers.Layer):
    """
    Prediction head for future poses and motion
    """
    def __init__(self, 
                 num_future_frames=5,
                 num_joints=18,
                 prediction_horizons=[1, 3, 5],
                 **kwargs):
        super().__init__(**kwargs)
        self.num_future_frames = num_future_frames
        self.num_joints = num_joints
        self.prediction_horizons = prediction_horizons
        
        # Multi-horizon prediction heads
        self.prediction_heads = {}
        for horizon in prediction_horizons:
            self.prediction_heads[f'horizon_{horizon}'] = tf.keras.Sequential([
                Dense(512, activation='relu'),
                Dense(256, activation='relu'),
                Dense(horizon * num_joints * 3, activation='linear'),
                Reshape((horizon, num_joints, 3))
            ], name=f'prediction_head_{horizon}')
        
        # Uncertainty estimation
        self.uncertainty_head = Dense(num_future_frames, activation='softplus', name='uncertainty')
        
        # Motion confidence
        self.confidence_head = Dense(num_future_frames, activation='sigmoid', name='confidence')
    
    def call(self, motion_encoding):
        """
        motion_encoding: (batch, latent_dim)
        Returns: dict with predictions for different horizons
        """
        predictions = {}
        
        # Multi-horizon predictions
        for horizon in self.prediction_horizons:
            pred_key = f'future_{horizon}_frames'
            predictions[pred_key] = self.prediction_heads[f'horizon_{horizon}'](motion_encoding)
        
        # Uncertainty and confidence
        predictions['uncertainty'] = self.uncertainty_head(motion_encoding)
        predictions['confidence'] = self.confidence_head(motion_encoding)
        
        return predictions


class HierarchicalTemporalAutoencoder(tf.keras.Model):
    """
    Main temporal prediction model extending hierarchical autoencoder
    """
    def __init__(self,
                 sequence_length=8,
                 prediction_horizons=[1, 3, 5],
                 latent_dim=512,
                 num_joints=18,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        
        # Temporal encoders
        self.tof_encoder = PhysicsInformedToFEncoder(latent_dim // 3)
        self.visual_encoder = VisualEncoder(latent_dim // 3) 
        self.motion_encoder = MotionDynamicsEncoder(latent_dim // 3)
        
        # Temporal hash encoding for 4D space-time
        self.temporal_hash_encoder = TemporalHashEncoding(
            num_levels=12,
            features_per_level=8,
            base_resolution=8,
            finest_resolution=256,
            temporal_resolution=sequence_length
        )
        
        # Temporal fusion
        self.temporal_fusion = LSTM(latent_dim, return_sequences=True, name='temporal_fusion')
        self.sequence_attention = MultiHeadAttention(num_heads=8, key_dim=64, name='sequence_attention')
        
        # Prediction head
        self.prediction_head = TemporalPredictionHead(
            num_future_frames=max(prediction_horizons),
            num_joints=num_joints,
            prediction_horizons=prediction_horizons
        )
        
        # Current frame reconstruction (for training stability)
        self.current_frame_decoder = tf.keras.Sequential([
            Dense(512, activation='relu'),
            Dense(num_joints * 3, activation='linear'),
            Reshape((num_joints, 3))
        ], name='current_decoder')
    
    def call(self, inputs, training=None):
        """
        inputs: {
            'tof_sequence': (batch, seq_len, 4, 4, 100, 1),
            'visual_sequence': (batch, seq_len, H, W, 3),
            'pose_sequence': (batch, seq_len, num_joints, 3)  # for training
        }
        """
        batch_size = tf.shape(inputs['tof_sequence'])[0]
        seq_len = tf.shape(inputs['tof_sequence'])[1]
        
        # Encode each frame in the sequence
        sequence_encodings = []
        
        for t in range(seq_len):
            frame_encodings = []
            
            # ToF encoding for frame t
            if 'tof_sequence' in inputs:
                tof_frame = inputs['tof_sequence'][:, t]
                tof_encoding = self.tof_encoder(tof_frame)
                frame_encodings.append(tof_encoding)
            
            # Visual encoding for frame t
            if 'visual_sequence' in inputs:
                visual_frame = inputs['visual_sequence'][:, t]
                visual_encoding = self.visual_encoder(visual_frame)
                frame_encodings.append(visual_encoding)
            
            # Combine modalities for this frame
            if len(frame_encodings) > 1:
                combined_encoding = tf.concat(frame_encodings, axis=-1)
            else:
                combined_encoding = frame_encodings[0]
            
            sequence_encodings.append(combined_encoding)
        
        # Stack sequence encodings
        sequence_tensor = tf.stack(sequence_encodings, axis=1)  # (batch, seq_len, latent_dim)
        
        # Motion dynamics encoding (if poses available)
        motion_encoding = None
        attention_weights = None
        if 'pose_sequence' in inputs and training:
            motion_encoding, attention_weights = self.motion_encoder(inputs['pose_sequence'])
            
            # Integrate motion dynamics
            motion_broadcast = tf.expand_dims(motion_encoding, axis=1)
            motion_broadcast = tf.tile(motion_broadcast, [1, seq_len, 1])
            sequence_tensor = tf.concat([sequence_tensor, motion_broadcast], axis=-1)
        
        # Temporal processing with LSTM
        lstm_output = self.temporal_fusion(sequence_tensor)
        
        # Sequence attention
        attended_sequence, seq_attention = self.sequence_attention(
            lstm_output, lstm_output, return_attention_scores=True
        )
        
        # Global sequence representation (use last timestep)
        global_encoding = attended_sequence[:, -1, :]
        
        # Generate predictions
        predictions = self.prediction_head(global_encoding)
        
        # Current frame reconstruction for training stability
        current_reconstruction = self.current_frame_decoder(global_encoding)
        
        # Add current frame and attention info
        predictions['current_frame'] = current_reconstruction
        predictions['sequence_attention'] = seq_attention
        predictions['motion_attention'] = attention_weights
        predictions['global_encoding'] = global_encoding
        
        return predictions


def temporal_prediction_loss(y_true, y_pred, alpha_temporal=1.0, alpha_consistency=0.5):
    """
    Multi-horizon temporal prediction loss with motion consistency
    """
    total_loss = 0.0
    
    # Multi-horizon prediction losses
    for horizon in [1, 3, 5]:
        if f'future_{horizon}_frames' in y_pred and f'future_{horizon}_frames' in y_true:
            pred_key = f'future_{horizon}_frames'
            
            # MSE loss for this horizon
            horizon_loss = tf.reduce_mean(tf.square(y_true[pred_key] - y_pred[pred_key]))
            
            # Weight by prediction difficulty (longer horizon = higher weight)
            horizon_weight = horizon / 5.0
            total_loss += horizon_weight * horizon_loss
    
    # Current frame reconstruction loss
    if 'current_frame' in y_pred and 'current_frame' in y_true:
        current_loss = tf.reduce_mean(tf.square(y_true['current_frame'] - y_pred['current_frame']))
        total_loss += alpha_temporal * current_loss
    
    # Motion consistency loss (velocity should be smooth)
    for horizon in [1, 3, 5]:
        if f'future_{horizon}_frames' in y_pred:
            pred_sequence = y_pred[f'future_{horizon}_frames']
            if horizon > 1:
                # Compute velocities
                velocities = pred_sequence[:, 1:] - pred_sequence[:, :-1]
                # Penalize large velocity changes
                velocity_consistency = tf.reduce_mean(tf.square(velocities[:, 1:] - velocities[:, :-1]))
                total_loss += alpha_consistency * velocity_consistency
    
    return total_loss


def create_temporal_prediction_model(config):
    """Factory function for temporal prediction model"""
    model = HierarchicalTemporalAutoencoder(
        sequence_length=config.get('sequence_length', 8),
        prediction_horizons=config.get('prediction_horizons', [1, 3, 5]),
        latent_dim=config.get('latent_dim', 512),
        num_joints=config.get('num_joints', 18)
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.get('lr', 0.0005)),
        loss=temporal_prediction_loss,
        metrics=['mae']
    )
    
    return model


if __name__ == "__main__":
    # Test temporal prediction
    config = {
        'sequence_length': 8,
        'prediction_horizons': [1, 3, 5],
        'latent_dim': 256,
        'num_joints': 18,
        'lr': 0.0005
    }
    
    model = create_temporal_prediction_model(config)
    
    # Test with dummy sequence data
    batch_size = 2
    seq_len = 8
    
    dummy_inputs = {
        'tof_sequence': tf.random.normal((batch_size, seq_len, 4, 4, 100, 1)),
        'visual_sequence': tf.random.normal((batch_size, seq_len, 64, 64, 3)),
        'pose_sequence': tf.random.normal((batch_size, seq_len, 18, 3))
    }
    
    outputs = model(dummy_inputs)
    
    print("Temporal Prediction Model Test:")
    print(f"✓ 1-frame prediction shape: {outputs['future_1_frames'].shape}")
    print(f"✓ 3-frame prediction shape: {outputs['future_3_frames'].shape}")
    print(f"✓ 5-frame prediction shape: {outputs['future_5_frames'].shape}")
    print(f"✓ Uncertainty shape: {outputs['uncertainty'].shape}")
    print(f"✓ Confidence shape: {outputs['confidence'].shape}")
    print("🎯 Temporal prediction ready!")
"""
Enhanced Pixels2Pose with Hierarchical Inductive Autoencoder
Integration layer for the new architecture with existing pipeline
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
import argparse
import time
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm

from hierarchical_inductive_autoencoder import create_hierarchical_inductive_model
from functions_FINALRESULTS import extract_parts, draw_3d, function_scale_color
from config_reader import config_reader
import util


class EnhancedPixels2Pose:
    """
    Enhanced Pixels2Pose system with hierarchical inductive autoencoder
    """
    def __init__(self, config=None):
        self.config = config or self._default_config()
        
        # Initialize models
        self.hierarchical_model = None
        self.load_models()
        
        # Processing parameters
        self.input_image_shape_0 = 200
        self.params, self.model_params = config_reader()
        
    def _default_config(self):
        """Default configuration for the hierarchical model"""
        return {
            'latent_dim': 512,
            'hierarchy_levels': 4,
            'output_resolution': (32, 32),
            'lr': 0.0005,
            'use_visual_fusion': True,
            'use_physics_constraints': True
        }
    
    def load_models(self):
        """Load the hierarchical inductive autoencoder"""
        print("Loading Hierarchical Inductive Autoencoder...")
        self.hierarchical_model = create_hierarchical_inductive_model(self.config)
        print("✓ Model loaded successfully")
    
    def load_pretrained_weights(self, weights_path):
        """Load pretrained weights if available"""
        if os.path.exists(weights_path):
            print(f"Loading pretrained weights from {weights_path}")
            self.hierarchical_model.load_weights(weights_path)
            return True
        return False
    
    def preprocess_tof_data(self, histogram_data):
        """
        Preprocess ToF histogram data for the hierarchical model
        """
        # Ensure correct shape: (batch, 4, 4, 100, 1)
        if len(histogram_data.shape) == 3:
            histogram_data = np.expand_dims(histogram_data, axis=0)
        if len(histogram_data.shape) == 4:
            histogram_data = np.expand_dims(histogram_data, axis=-1)
            
        # Normalize histogram data
        histogram_data = histogram_data.astype(np.float32)
        histogram_data = (histogram_data - histogram_data.min()) / (histogram_data.max() - histogram_data.min() + 1e-8)
        
        return histogram_data
    
    def preprocess_visual_data(self, rgb_image):
        """
        Preprocess visual data for the hierarchical model
        """
        if rgb_image is None:
            return None
            
        # Resize to appropriate size
        visual_size = self.config.get('visual_input_size', (128, 128))
        rgb_processed = cv2.resize(rgb_image, visual_size)
        
        # Normalize
        rgb_processed = rgb_processed.astype(np.float32) / 255.0
        
        # Add batch dimension
        if len(rgb_processed.shape) == 3:
            rgb_processed = np.expand_dims(rgb_processed, axis=0)
            
        return rgb_processed
    
    def predict_hierarchical(self, tof_data, visual_data=None):
        """
        Run inference with hierarchical inductive autoencoder
        """
        # Prepare inputs
        inputs = {'tof': tof_data}
        
        if visual_data is not None and self.config['use_visual_fusion']:
            inputs['visual'] = visual_data
        
        # Run inference
        start_time = time.time()
        outputs = self.hierarchical_model(inputs, training=False)
        inference_time = time.time() - start_time
        
        return outputs, inference_time
    
    def extract_poses_from_hierarchical_output(self, hierarchical_output, size_image):
        """
        Extract 3D poses from hierarchical model output
        """
        # Get pose heatmaps and depth
        pose_heatmaps = hierarchical_output['pose']  # (batch, 32, 32, 18)
        depth_map = hierarchical_output['depth']     # (batch, 32, 32, 1) or similar
        
        # Resize to match expected input size
        if pose_heatmaps.shape[1:3] != size_image:
            pose_heatmaps_resized = tf.image.resize(
                pose_heatmaps, size_image, method='bilinear'
            ).numpy()
            depth_map_resized = tf.image.resize(
                depth_map, size_image, method='bilinear'
            ).numpy()
        else:
            pose_heatmaps_resized = pose_heatmaps.numpy()
            depth_map_resized = depth_map.numpy()
        
        # Extract pose information using existing pipeline
        batch_size = pose_heatmaps_resized.shape[0]
        all_poses_3d = []
        
        for batch_idx in range(batch_size):
            # Get single sample
            heatmap_sample = pose_heatmaps_resized[batch_idx]  # (H, W, 18)
            depth_sample = depth_map_resized[batch_idx, :, :, 0] * 250  # Convert to mm
            
            # Create PAFs (Part Affinity Fields) - simplified version
            # In practice, you might want to train the model to output PAFs directly
            paf_sample = np.zeros((*heatmap_sample.shape[:2], 38))  # 38 PAF channels
            
            # Extract body parts using existing function
            body_parts, all_peaks, subset_est, candidate = extract_parts(
                heatmap_sample, paf_sample, self.input_image_shape_0, 
                self.params, self.model_params
            )
            
            # Convert to 3D using depth information
            candidate_3D_estimated, lines_est, corresponding_color = draw_3d(
                1, depth_sample, subset_est, candidate
            )
            
            all_poses_3d.append({
                'candidate_3D': candidate_3D_estimated,
                'lines': lines_est,
                'colors': corresponding_color,
                'subset': subset_est,
                'candidate': candidate
            })
        
        return all_poses_3d
    
    def run_enhanced_pipeline(self, scenario_path, use_visual=True):
        """
        Run the complete enhanced pipeline
        """
        print(f"Running Enhanced Pixels2Pose Pipeline...")
        print(f"Scenario path: {scenario_path}")
        
        # Load data
        histogram_validation = sio.loadmat(os.path.join(scenario_path, 'data.mat'))['histogram']
        rgb_image = sio.loadmat(os.path.join(scenario_path, 'data.mat'))['reference_RGB']
        
        print(f"Loaded histogram shape: {histogram_validation.shape}")
        print(f"Loaded RGB shape: {rgb_image.shape}")
        
        # Preprocess data
        tof_data = self.preprocess_tof_data(histogram_validation)
        visual_data = self.preprocess_visual_data(rgb_image) if use_visual else None
        
        # Run hierarchical inference
        hierarchical_outputs, inference_time = self.predict_hierarchical(tof_data, visual_data)
        
        print(f"Hierarchical inference time: {inference_time:.3f}s")
        print(f"Depth output shape: {hierarchical_outputs['depth'].shape}")
        print(f"Pose output shape: {hierarchical_outputs['pose'].shape}")
        
        # Extract 3D poses
        size_image = (rgb_image.shape[0], rgb_image.shape[1])
        poses_3d = self.extract_poses_from_hierarchical_output(hierarchical_outputs, size_image)
        
        # Visualization
        self.visualize_results(
            rgb_image, 
            hierarchical_outputs['depth'].numpy()[0],
            poses_3d[0],
            scenario_path
        )
        
        return hierarchical_outputs, poses_3d, inference_time
    
    def visualize_results(self, rgb_image, depth_map, pose_3d, output_path):
        """
        Visualize results similar to original pipeline
        """
        viridis = cm.get_cmap('viridis')
        fig = plt.figure(figsize=(18, 6))
        
        # RGB reference
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(rgb_image)
        ax1.set_title('Reference RGB image', fontsize=16)
        ax1.axis('off')
        
        # Depth map
        ax2 = fig.add_subplot(1, 3, 2)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        
        # Resize depth map to match RGB
        depth_resized = cv2.resize(depth_map.squeeze(), 
                                 (rgb_image.shape[1], rgb_image.shape[0]))
        
        im = ax2.imshow(depth_resized)
        im.set_clim(0, 1)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical', 
                           ticks=[0.33, 0.66, 0.99])
        cbar.ax.set_yticklabels(['1m', '2m', '3m'], fontsize=12)
        ax2.set_title('Enhanced Depth Output', fontsize=16)
        ax2.axis('off')
        
        # 3D pose
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        
        # Draw 3D skeleton if we have valid poses
        if 'candidate_3D' in pose_3d and pose_3d['candidate_3D'] is not None:
            candidate_3D = pose_3d['candidate_3D']
            lines_est = pose_3d['lines']
            corresponding_color = pose_3d['colors']
            
            # Draw joints
            for i, point in enumerate(candidate_3D):
                if len(point) >= 3:
                    x, y, z = point[0], point[1], point[2]
                    val = function_scale_color(z / 300)
                    color_depth = viridis(val)
                    ax3.scatter(y, z, rgb_image.shape[0] - x, s=100, color=color_depth)
            
            # Draw connections
            if lines_est is not None and len(lines_est) > 0:
                c = viridis(function_scale_color(corresponding_color))
                lc = Line3DCollection(lines_est, linewidths=3, colors=c)
                ax3.add_collection3d(lc)
        
        # Set 3D axes
        ax3.set_xlim3d(0, 200)
        ax3.set_ylim3d(0, 300)
        ax3.set_zlim3d(0, rgb_image.shape[0])
        ax3.set_xlabel('y (m)', fontsize=12)
        ax3.set_ylabel('z (m)', fontsize=12)
        ax3.set_zlabel('x (m)', fontsize=12)
        ax3.set_title('Enhanced 3D Pose', fontsize=16)
        ax3.view_init(10, -60)
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_path, 'enhanced_result.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Results saved to: {output_file}")
        
        plt.show()
    
    def train_hierarchical_model(self, training_data, validation_data, epochs=100):
        """
        Train the hierarchical inductive autoencoder
        """
        print("Training Hierarchical Inductive Autoencoder...")
        
        # Prepare callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'hierarchical_model_best.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            )
        ]
        
        # Train model
        history = self.hierarchical_model.fit(
            training_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history


def main():
    parser = argparse.ArgumentParser(description='Enhanced Pixels2Pose with Hierarchical Autoencoder')
    parser.add_argument('--scenario', type=str, required=True, 
                       help='Scenario number (1, 2, or 3)')
    parser.add_argument('--use_visual', type=bool, default=True,
                       help='Use visual fusion')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--weights_path', type=str, default=None,
                       help='Path to pretrained weights')
    
    args = parser.parse_args()
    
    # Set up paths
    scenario_path = os.path.join(os.getcwd(), f"{args.scenario}_PEOPLE")
    
    if not os.path.exists(scenario_path):
        print(f"Error: Scenario path {scenario_path} does not exist!")
        return
    
    # Initialize enhanced system
    enhanced_system = EnhancedPixels2Pose()
    
    # Load pretrained weights if provided
    if args.weights_path and os.path.exists(args.weights_path):
        enhanced_system.load_pretrained_weights(args.weights_path)
    
    # Run enhanced pipeline
    start_time = time.time()
    hierarchical_outputs, poses_3d, inference_time = enhanced_system.run_enhanced_pipeline(
        scenario_path, use_visual=args.use_visual
    )
    total_time = time.time() - start_time
    
    print(f"\n=== Enhanced Pixels2Pose Results ===")
    print(f"Total processing time: {total_time:.3f}s")
    print(f"Inference time: {inference_time:.3f}s")
    print(f"Preprocessing + postprocessing time: {total_time - inference_time:.3f}s")
    print(f"Estimated FPS: {1.0 / total_time:.2f}")
    
    print(f"\n=== Model Architecture Benefits ===")
    print("✓ Multi-resolution hash encoding for efficient spatial representation")
    print("✓ Physics-informed ToF processing with inductive biases")
    print("✓ Hierarchical processing for multi-scale feature extraction")
    print("✓ Multi-modal fusion (ToF + Visual)")
    print("✓ End-to-end differentiable architecture")


if __name__ == "__main__":
    main()
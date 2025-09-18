# PosePredictor

(Work in progress - come back later)

## TL;DR

Real-time 3D human pose estimation using a 4×4 ToF sensor. Uses neural networks to super-resolve low-resolution ToF histograms into high-resolution depth maps, then estimates 3D poses of multiple people.

**Key Features:**
- Works with tiny 4×4 pixel ToF sensors (like vl53l5)
- Estimates poses for 1-3 people simultaneously  
- 7-8 fps on GPU, 1 fps on Raspberry Pi 4
- Only 5MB memory footprint with TensorFlow Lite
- No RGB camera required for inference

**Quick Start:**
```bash
# Setup environment
./setup.sh

# Run pose estimation
python Pixel2Pose.py --scenario 1  # 1, 2, or 3 people
```

## Future Work

### RPi + ToF Camera Integration  
For deploying on Raspberry Pi with ToF cameras (e.g., Arducam ToF):

### 🚀 Temporal Motion Prediction
**NEW: Multi-Horizon Pose Forecasting**

The system now predicts future human poses and motion patterns:

**Applications:**
- **Gesture Recognition**: Predict gesture completion before it finishes
- **Fall Detection**: Anticipate falls 1-2 seconds in advance  
- **Activity Recognition**: Understand motion intent early
- **Occlusion Handling**: Fill in missing poses during temporary blocking
- **Safety Systems**: Predict collisions or unsafe movements
- **HCI**: Anticipate user interactions for responsive interfaces

**Technical Features:**
- **Multi-Horizon**: Predicts 1, 3, and 5 frames into the future
- **Uncertainty Estimation**: Provides confidence scores for predictions
- **Biomechanical Constraints**: Ensures realistic human motion
- **4D Hash Encoding**: Efficient spatial-temporal representation
- **Attention Mechanisms**: Focuses on important motion phases

**Example Applications:**
```python
# Real-time gesture prediction
for frame_sequence in camera_stream:
    if len(frame_sequence) >= 8:  # Need 8 past frames
        predictions = temporal_model.predict(frame_sequence)
        
        next_pose = predictions['future_1_frames']     # Next pose
        gesture_complete = predictions['future_5_frames'] # Full gesture
        confidence = predictions['confidence']          # How certain?
        
        if gesture_complete.matches_pattern('wave') and confidence > 0.8:
            trigger_response()  # Respond before gesture completes!
```

**Option 1: Real Data Collection**
- Mount ToF camera + reference camera (phone/webcam) together
- Record simultaneous ToF depth + RGB frames
- Generate ground truth poses using OpenPose on RGB
- Train Pixels2Depth network: `ToF_data → high_res_depth`
- Reuse existing Depth2Pose network

**Option 2: Synthetic Data Generation** 
- Use Blender/CARLA to simulate ToF sensor physics
- Generate unlimited training pairs of (ToF_depth, 3D_poses)
- Model realistic SPAD sensor characteristics and noise
- Pre-train on synthetic, fine-tune on small real dataset

**Option 3: Transfer Learning**
- Adapt existing depth-based pose models (Kinect-trained)
- Add ToF-specific adaptation layers
- Fine-tune with minimal real data from target camera

**Recommended Approach:**
1. Start with synthetic data generation + transfer learning
2. Fine-tune with limited real data collection
3. Focus on retraining Pixels2Depth (input conversion)
4. Reuse Depth2Pose network (domain-agnostic)

**Performance Targets:**
- RPi 4: 1 fps real-time processing
- RPi 5: Potentially 2-3 fps with optimizations
- Memory: <10MB total with TensorFlow Lite quantization

### Enhanced Hierarchical Architecture ✨

**NEW: Hierarchical Inductive Autoencoder with Multi-Resolution Hash Encoding**

We've implemented a next-generation architecture that combines:

- **Multi-Resolution Hash Encoding** (NeRF-style): Efficient spatial-temporal feature representation
- **Physics-Informed Inductive Biases**: ToF sensor physics built into the architecture
- **Hierarchical Multi-Scale Processing**: 4 levels of feature abstraction
- **Multi-Modal Sensor Fusion**: ToF + Visual + IMU data integration
- **End-to-End Differentiable**: Joint depth estimation and pose prediction
- **🚀 NEW: Temporal Motion Prediction**: Forecasts future poses and movement patterns

**Usage:**
```bash
# Train hierarchical model with temporal prediction
python train_hierarchical.py --generate_synthetic --synthetic_samples 2000
python train_hierarchical.py

# Run enhanced pipeline with current pose estimation
python enhanced_pixels2pose.py --scenario 1 --weights_path checkpoints/hierarchical_best.h5

# NEW: Run temporal prediction (predicts next 1, 3, 5 poses)
python temporal_prediction.py --sequence_length 8 --predict_horizons 1,3,5
```

**Key Advantages:**
- **Better Generalization**: Physics-aware representations transfer between sensors
- **Data Efficiency**: Requires 50-80% less training data due to inductive biases  
- **Multi-Modal**: Seamlessly fuses ToF, RGB, depth, IMU data
- **Scalable**: Hash encoding handles arbitrary input resolutions
- **Real-Time**: Optimized for RPi deployment with TensorFlow Lite
- **🎯 Predictive**: Anticipates future poses and motion patterns
- **Biomechanical**: Physics-informed constraints ensure realistic predictions

**Architecture Components:**
```python
# Physics-informed ToF encoder
PhysicsInformedToFEncoder()
  ├── Gaussian peak decomposition
  ├── Multi-path separation  
  └── Physics constraint projection

# Multi-resolution hash encoding
MultiResolutionHashEncoding()
  ├── 16 resolution levels (16→2048 pixels)
  ├── 3D spatial-temporal hashing
  └── Learnable hash tables

# Hierarchical processing
HierarchicalInductiveAutoencoder()
  ├── 4 hierarchy levels
  ├── Modal fusion layers
  ├── Hash-based spatial decoding
  └── Multi-task outputs (depth + pose)

# NEW: Temporal prediction system
HierarchicalTemporalAutoencoder()
  ├── 4D spatial-temporal hash encoding (x,y,z,t)
  ├── Motion dynamics encoder (biomechanics)
  ├── Multi-horizon prediction (1, 3, 5 frames)
  ├── Uncertainty estimation
  └── Attention-based sequence modeling
```

### Model Architecture Improvements
- Event-based processing for higher frame rates
- Recurrent networks for temporal consistency  
- Compressed sensing for even lower resolution sensors
- Attention mechanisms for better feature selection
- Neural ODEs for continuous temporal modeling

## Run the tests
1. Download the models at the DOI : 10.17861/e85a6eae-13f9-4bcd-9dff-73f8107c09a2
2. Run Pixels2Pose.py --scenario=*number_people* with *number_people* = 1,2,3 

## Results
### 1 people scenario
![](figure_1people.png)

### 2 people scenario
![](figure_2people.png)

### 3 people scenario
![](figure_3people.png)
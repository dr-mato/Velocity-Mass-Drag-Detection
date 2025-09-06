# Ball Motion Analysis & Parameter Estimation

A computer vision system that tracks ball motion in videos and estimates physical parameters (mass, drag coefficient, rolling resistance) using numerical optimization and physics simulation.

## Overview

This project combines computer vision techniques with physics modeling to analyze ball motion in videos and extract real-world physical parameters. The system tracks a ball's trajectory, calculates velocities, and uses numerical optimization to estimate the ball's mass, drag coefficient, and rolling resistance based on observed motion patterns.

## Features

- **Video Ball Tracking**: Automated ball position tracking using background subtraction
- **Physics Simulation**: Euler integration with drag and rolling resistance modeling  
- **Parameter Estimation**: Numerical optimization to fit physical parameters to observed motion
- **Unit Conversion**: Pixel-to-meter conversion for real-world measurements
- **Velocity Calculation**: Automatic velocity computation from position data
- **Comparative Analysis**: Testing with both high-quality and poor-quality tracking scenarios

## Key Functions

### `track_ball(video_path)`
Tracks the ball's position throughout the video using background subtraction techniques.
- **Input**: Path to video file
- **Output**: Time points, x/y coordinates, frame rate, frame size, ball radii

### `euler_step(y, t, dt, drag_coeff, rolling_coeff, mass, g)`
Performs Euler integration step for ball motion with drag and rolling resistance.
- **Physics Model**: Incorporates gravity, air drag, and surface rolling resistance
- **Output**: Next position and velocity state

### `objective_function(params, ...)`
Optimization objective function that compares observed vs. simulated trajectories.
- **Method**: Sum of squared differences between observed and calculated positions
- **Purpose**: Parameter fitting for mass and drag coefficient estimation

### `estimate_parameters(time_points, x_positions, y_positions, ...)`
Estimates physical parameters using numerical optimization methods.
- **Algorithm**: Minimizes difference between observed and simulated motion
- **Output**: Estimated mass, drag coefficient, and rolling resistance

### `convert_pixels_to_meters(pixels, real_length_meters, pixels_length)`
Converts pixel measurements to real-world metric units.
- **Calibration**: Uses known object dimensions for scale conversion

### `calculate_velocities(time_points, x_positions, y_positions, ...)`
Computes ball velocities from position data using numerical differentiation.

## Results & Validation

### High-Quality Tracking (Football)
- **Estimated Mass**: 0.4695 kg
- **Estimated Drag Coefficient**: 0.3942  
- **Estimated Rolling Coefficient**: 4.0755

### Poor-Quality Tracking (Golf Ball)
- **Estimated Mass**: 0.3427 kg
- **Estimated Drag Coefficient**: 0.3999
- **Estimated Rolling Coefficient**: 2.8763

### Analysis
The football results show good accuracy for mass estimation (~0.47 kg is realistic for a football). The golf ball results are less accurate due to:
- Poor tracking quality in the video
- Suboptimal camera angle
- Limited visible motion making parameter estimation challenging

## Technical Approach

1. **Computer Vision**: Background subtraction for ball detection and tracking
2. **Physics Modeling**: Ordinary differential equation (ODE) integration with:
   - Gravitational acceleration
   - Air drag forces  
   - Rolling resistance on surfaces
3. **Numerical Optimization**: Parameter fitting using least-squares minimization
4. **Validation**: Comparison between high-quality and poor-quality tracking scenarios

## Requirements

```python
opencv-python
numpy
scipy
matplotlib
```

## Usage

```python
# Track ball in video
time_points, x_pos, y_pos, fps, frame_size, radii = track_ball('video.mp4')

# Convert to real-world units
x_meters = convert_pixels_to_meters(x_pos, real_length, pixel_length)
y_meters = convert_pixels_to_meters(y_pos, real_length, pixel_length)

# Calculate velocities
vx, vy = calculate_velocities(time_points, x_meters, y_meters, radii, ball_radius)

# Estimate physical parameters
mass, drag_coeff = estimate_parameters(time_points, x_meters, y_meters, 
                                     initial_state, gravity, vx, vy)
```

## Applications

- **Sports Analysis**: Ball trajectory analysis for performance optimization
- **Physics Education**: Demonstrating real-world physics concepts
- **Computer Vision Research**: Object tracking and parameter estimation
- **Robotics**: Understanding projectile motion for catching/throwing systems

## Limitations

- Tracking quality heavily affects parameter estimation accuracy
- Requires clear ball visibility and contrast with background
- Camera angle and perspective influence measurement precision
- Works best with substantial visible motion for reliable parameter fitting

## Future Improvements

- Enhanced tracking algorithms (e.g., Kalman filtering, deep learning-based detection)
- Multi-camera setups for 3D trajectory analysis  
- Real-time processing capabilities
- Advanced physics models (spin, air pressure effects)
- Automated calibration methods

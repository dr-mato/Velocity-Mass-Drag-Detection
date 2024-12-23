import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def track_ball(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error opening video file")

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    time_step = 1 / frame_rate
    positions_x = []
    positions_y = []
    time_points = []
    radii = []
    frame_count = 0
    
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        fgmask = fgbg.apply(frame) # Apply the background subtractor
        
        # Remove noise
        fgmask = cv2.medianBlur(fgmask, 5)
        fgmask = cv2.erode(fgmask, None, iterations=2)
        fgmask = cv2.dilate(fgmask, None, iterations=2)
        
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
          largest_contour = max(contours, key=cv2.contourArea)
          ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
          if radius > 10:
              center = (int(x), int(y))
              positions_x.append(center[0])
              positions_y.append(center[1])
              time_points.append(time_step * frame_count)
              radii.append(radius)
              cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
              
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    return time_points, np.array(positions_x), np.array(positions_y), frame_rate, frame_size, np.array(radii)

def euler_step(y, t, dt, drag_coeff, rolling_coeff, mass, g):
    pos_x, pos_y, vel_x, vel_y = y
    speed = np.sqrt(vel_x**2 + vel_y**2)
    vel_x = min(vel_x, 100)
    vel_x = max(vel_x, -100)
    vel_y = min(vel_y, 100)
    vel_y = max(vel_y, -100)
    speed = min(speed, 100) # Clamp speed to a maximum of 100.
    
    # Air drag force
    drag_force_x = -drag_coeff * speed * vel_x
    drag_force_y = -drag_coeff * speed * vel_y
    
    # Rolling resistance force
    rolling_force_magnitude = rolling_coeff*mass*g if speed > 0.01 else 0
    rolling_force_x = -rolling_force_magnitude * (vel_x / (speed + 0.001)) if speed > 0.01 else 0
    rolling_force_y = -rolling_force_magnitude * (vel_y / (speed + 0.001)) if speed > 0.01 else 0
    
    
    a_x = (drag_force_x + rolling_force_x) / mass
    a_y = (drag_force_y + rolling_force_y) / mass - g
    
    vel_x_next = vel_x + a_x*dt
    vel_y_next = vel_y + a_y*dt
    pos_x_next = pos_x + vel_x*dt
    pos_y_next = pos_y + vel_y*dt
    return np.array([pos_x_next, pos_y_next, vel_x_next, vel_y_next])
    
def objective_function(params, time_points, initial_state, g, observed_positions_x,observed_positions_y, observed_vx, observed_vy):
    mass, drag_coeff, rolling_coeff = params
    
    error_x = 0
    error_y = 0

    for i in range(len(observed_vx)-1):
        
        # Air drag force
        speed = np.sqrt(observed_vx[i]**2 + observed_vy[i]**2)
        drag_force_x = -drag_coeff * speed * observed_vx[i]
        drag_force_y = -drag_coeff * speed * observed_vy[i]
    
        # Rolling resistance force
        rolling_force_magnitude = rolling_coeff*mass*g if speed > 0.01 else 0
        rolling_force_x = -rolling_force_magnitude * (observed_vx[i] / (speed + 0.001)) if speed > 0.01 else 0
        rolling_force_y = -rolling_force_magnitude * (observed_vy[i] / (speed + 0.001)) if speed > 0.01 else 0
    
        a_x = (drag_force_x + rolling_force_x) / mass
        a_y = (drag_force_y + rolling_force_y) / mass - g
        
        a_x_observed = (observed_vx[i+1]-observed_vx[i])/(time_points[i+1]-time_points[i])
        a_y_observed = (observed_vy[i+1]-observed_vy[i])/(time_points[i+1]-time_points[i])
        
        error_x += (a_x - a_x_observed)**2
        error_y += (a_y- a_y_observed)**2
    return error_x+error_y

def estimate_parameters(time_points, x_positions, y_positions, initial_state, g, vx, vy):
    initial_guess = np.array([0.3, 0.4, 0.01]) # initial guess on mass, and drag coefficient, rolling coefficient
    result = minimize(objective_function, initial_guess, args=(time_points,initial_state,g,x_positions,y_positions,vx,vy),method='Powell')
    mass, drag_coeff, rolling_coeff = result.x
    return mass, drag_coeff, rolling_coeff

def convert_pixels_to_meters(pixels, real_length_meters, pixels_length):
    return (pixels / pixels_length) * real_length_meters


def calculate_velocities(time_points, x_positions, y_positions, radii, ball_radius_meters):
  x_positions_meters = convert_pixels_to_meters(x_positions,ball_radius_meters*2,radii*2)
  y_positions_meters = convert_pixels_to_meters(y_positions,ball_radius_meters*2,radii*2)

  # Calculate velocities, skipping invalid frames.
  vx = []
  vy = []
  valid_time_points = []
  for i in range(len(x_positions_meters)-1):
      if abs(radii[i+1]-radii[i]) < radii[i]*0.5:
        vx.append((x_positions_meters[i+1]-x_positions_meters[i])/(time_points[i+1]-time_points[i]))
        vy.append((y_positions_meters[i+1]-y_positions_meters[i])/(time_points[i+1]-time_points[i]))
        valid_time_points.append(time_points[i])
    
  return np.array(vx), np.array(vy), np.array(valid_time_points)


def main():
    # --- Video Tracking (Good Example) ---
    video_path = 'ball_moving.mp4'
    time_points, x_positions, y_positions, frame_rate, frame_size, radii = track_ball(video_path)

    if time_points is None or x_positions is None or y_positions is None:
      print("Error: Failed to track the ball.")
      return
    
    if len(time_points) < 2:
      print("Error: Ball detected in too few frames to calculate velocities")
      return

    # Calculate velocities
    ball_radius_meters = 0.12
    if len(radii) > 0:
      radii = np.pad(radii,(0,1), mode='edge')
    vx, vy, time_points_vel = calculate_velocities(time_points, x_positions, y_positions, radii[:-1], ball_radius_meters)

    if len(vx) < 1:
        print("Error: Insufficient frames to calculate velocity.")
        return
        
    initial_state = np.array([x_positions[0], y_positions[0], vx[0], vy[0]])
    g = 9.81
    
    mass, drag_coeff, rolling_coeff = estimate_parameters(time_points, x_positions, y_positions, initial_state, g, vx, vy)
    print(f"Estimated Mass: {mass:.4f} kg")
    print(f"Estimated Drag Coefficient: {drag_coeff:.4f}")
    print(f"Estimated Rolling Coefficient: {rolling_coeff:.4f}")
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(time_points_vel, vx, marker='o', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity X (m/s)')
    plt.title('Ball Velocity vs. Time (Good)')

    plt.subplot(1, 2, 2)
    plt.plot(time_points_vel, vy, marker='o', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity Y (m/s)')
    plt.title('Ball Velocity vs. Time (Good)')

    plt.tight_layout()
    plt.show()
    
    # --- Video Tracking (Bad Example) ---
    video_path_bad = 'bad_shot.mp4'
    time_points_bad, x_positions_bad, y_positions_bad, frame_rate_bad, frame_size_bad, radii_bad = track_ball(video_path_bad)

    if time_points_bad is None or x_positions_bad is None or y_positions_bad is None:
      print("Error: Failed to track the ball.")
      return
    
    if len(time_points_bad) < 2:
      print("Error: Ball detected in too few frames to calculate velocities")
      return

    # Calculate velocities
    ball_radius_meters = 0.03
    if len(radii_bad) > 0:
      radii_bad = np.pad(radii_bad,(0,1), mode='edge')
    vx_bad, vy_bad, time_points_vel_bad = calculate_velocities(time_points_bad, x_positions_bad, y_positions_bad, radii_bad[:-1], ball_radius_meters)

    if len(vx_bad) < 1:
        print("Error: Insufficient frames to calculate velocity.")
        return
        
    initial_state_bad = np.array([x_positions_bad[0], y_positions_bad[0], vx_bad[0], vy_bad[0]])
    g = 9.81
    
    mass_bad, drag_coeff_bad, rolling_coeff_bad = estimate_parameters(time_points_bad, x_positions_bad, y_positions_bad, initial_state_bad, g, vx_bad, vy_bad)
    print(f"Estimated Mass (Bad Example): {mass_bad:.4f} kg")
    print(f"Estimated Drag Coefficient (Bad Example): {drag_coeff_bad:.4f}")
    print(f"Estimated Rolling Coefficient (Bad Example): {rolling_coeff_bad:.4f}")
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(time_points_vel, vx, marker='o', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity X (m/s)')
    plt.title('Ball Velocity vs. Time')

    plt.subplot(1, 2, 2)
    plt.plot(time_points_vel, vy, marker='o', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity Y (m/s)')
    plt.title('Ball Velocity vs. Time')

    plt.tight_layout()
    plt.show()
    
main()
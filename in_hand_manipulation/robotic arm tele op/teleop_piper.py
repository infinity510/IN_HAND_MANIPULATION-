import time
import cv2
import math
import numpy as np
from webcam_aruco_tracker import WebcamArucoTracker
from piper_sdk import *

def clip_pose_mm(x, y, z):
    """
    Clamps coordinates (in mm) to a safe workspace bounding box.
    Prevents the IK solver from crashing into the physical table or its own base.
    """
    x = np.clip(x, 150, 400)   # X: Forward/Backward (mm)
    y = np.clip(y, -250, 250)  # Y: Left/Right (mm)
    z = np.clip(z, 50, 350)    # Z: Up/Down (mm)
    return x, y, z

def main():
    print("Initializing Piper-X Arm on can0...")
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    
    print("Enabling arm (waiting for ready)...")
    while not piper.EnablePiper():
        time.sleep(0.01)
    print("Arm Enabled and Holding Position.")

    # Initialize gripper
    piper.GripperCtrl(0, 1000, 0x01, 0)

    print("Starting Webcam Tracker...")
    tracker = WebcamArucoTracker(marker_length=0.093, alpha=0.65, camera_index=1)
    tracker.start()
    
    print("\n=== SYSTEM READY ===")
    print("1. Manually move the physical arm near the center of your desk.")
    print("2. Hold the ArUco marker in the camera view.")
    print("3. Press 'o' in the video window to lock origin and begin tracking.")
    print("4. Press 'c' to toggle the gripper (Open / Soft Close).")
    print("5. Press 's' to pause movement. Press 'q' to quit.\n")
    
    is_active = False
    origin_pose = None
    error_printed = False
    gripper_is_open = False # Starts closed

    # Robot home position based on actual safe pose
    ROBOT_HOME_X_MM = 287.8    # Forward (mm)
    ROBOT_HOME_Y_MM = 17.9     # Left/Right (mm)
    ROBOT_HOME_Z_MM = 287.6    # Up (mm)
    ROBOT_HOME_RX_DEG = -90.4  # Roll (degrees)
    ROBOT_HOME_RY_DEG = 9.9    # Pitch (degrees)
    ROBOT_HOME_RZ_DEG = -92.5  # Yaw (degrees)

    # SDK unit factor: positions in 0.001mm, angles in 0.001deg
    FACTOR = 1000

    # Exponential Moving Average state for smoothing coordinates
    # Alpha controls smoothness. Lower = smoother but more delay. (0.0 to 1.0)
    # Changed from 0.15 to 0.05 for extremely smooth, vibration-free tracking
    ema_alpha = 0.05
    ema_target = None

    try:
        while True:
            start_time = time.time()
            positions, frame = tracker.step()
            
            # Handle Keyboard Toggles
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('o'):
                if positions and positions.get(0) is not None:
                    # 1. Capture Camera Origin
                    origin_pose = positions[0].copy()
                    
                    # 2. Capture Robot's Current Physical Pose as the new Home
                    current_robot_pose = piper.GetArmEndPoseMsgs()
                    ROBOT_HOME_X_MM = current_robot_pose.end_pose.X_axis / 1000.0
                    ROBOT_HOME_Y_MM = current_robot_pose.end_pose.Y_axis / 1000.0
                    ROBOT_HOME_Z_MM = current_robot_pose.end_pose.Z_axis / 1000.0
                    ROBOT_HOME_RX_DEG = current_robot_pose.end_pose.RX_axis / 1000.0
                    ROBOT_HOME_RY_DEG = current_robot_pose.end_pose.RY_axis / 1000.0
                    ROBOT_HOME_RZ_DEG = current_robot_pose.end_pose.RZ_axis / 1000.0
                    
                    is_active = True
                    error_printed = False
                    ema_target = None # Reset smoothing on new origin
                    print(f"\n>>> Origin locked! Robot Home dynamically set to: X={ROBOT_HOME_X_MM:.1f}mm, Y={ROBOT_HOME_Y_MM:.1f}mm, Z={ROBOT_HOME_Z_MM:.1f}mm <<<")
                    print("Teleoperation ACTIVE.")
                else:
                    print("Cannot set origin: ArUco marker (ID: 0) not currently visible.")
            elif key == ord('s'):
                is_active = False
                print("Teleoperation PAUSED.")
            elif key == ord('c'):
                gripper_is_open = not gripper_is_open
                if gripper_is_open:
                    print("Gripper: OPENING")
                    # Open to 70mm, normal torque
                    piper.GripperCtrl(70000, 1000, 0x01, 0)
                else:
                    print("Gripper: SOFT CLOSING (Grasping)")
                    # Close to 0mm, but limit torque to 800 (0.8 N.m) to grip without crushing
                    piper.GripperCtrl(0, 800, 0x01, 0)
            
            if frame is not None:
                if positions and positions.get(0) is not None:
                    current_raw_pose = positions[0]
                    
                    if is_active and origin_pose is not None:
                        # Calculate physical delta from the captured origin (in meters and radians)
                        dx_m = current_raw_pose[0] - origin_pose[0]
                        dy_m = current_raw_pose[1] - origin_pose[1]
                        dz_m = current_raw_pose[2] - origin_pose[2]
                        drx_rad = current_raw_pose[3] - origin_pose[3]
                        dry_rad = current_raw_pose[4] - origin_pose[4]
                        drz_rad = current_raw_pose[5] - origin_pose[5]
                        
                        # Convert deltas: meters -> mm, radians -> degrees
                        dx_mm = dx_m * 1000.0
                        dy_mm = dy_m * 1000.0
                        dz_mm = dz_m * 1000.0
                        drx_deg = math.degrees(drx_rad)
                        dry_deg = math.degrees(dry_rad)
                        drz_deg = math.degrees(drz_rad)
                        
                        # Apply delta to the robot's home position (in mm and degrees)
                        target_x_mm = ROBOT_HOME_X_MM + dx_mm
                        target_y_mm = ROBOT_HOME_Y_MM + dy_mm
                        target_z_mm = ROBOT_HOME_Z_MM + dz_mm
                        target_rx_deg = ROBOT_HOME_RX_DEG + drx_deg
                        target_ry_deg = ROBOT_HOME_RY_DEG + dry_deg
                        target_rz_deg = ROBOT_HOME_RZ_DEG + drz_deg
                        
                        # Enforce the safety bounding box (in mm)
                        target_x_mm, target_y_mm, target_z_mm = clip_pose_mm(
                            target_x_mm, target_y_mm, target_z_mm
                        )

                        # Apply Exponential Moving Average (EMA) to smooth out vibrations
                        current_target = np.array([
                            target_x_mm, target_y_mm, target_z_mm,
                            target_rx_deg, target_ry_deg, target_rz_deg
                        ])
                        
                        if ema_target is None:
                            ema_target = current_target
                        else:
                            ema_target = ema_alpha * current_target + (1.0 - ema_alpha) * ema_target
                            
                        sm_x, sm_y, sm_z, sm_rx, sm_ry, sm_rz = ema_target
                        
                        # Convert smoothed targets to SDK units: mm * 1000 and deg * 1000
                        x_sdk = round(sm_x * FACTOR)
                        y_sdk = round(sm_y * FACTOR)
                        z_sdk = round(sm_z * FACTOR)
                        rx_sdk = round(sm_rx * FACTOR)
                        ry_sdk = round(sm_ry * FACTOR)
                        rz_sdk = round(sm_rz * FACTOR)
                        
                        # CRITICAL: Must send MotionCtrl_2 AND EndPoseCtrl together every frame
                        try:
                            # move_spd_rate_ctrl changed from 100 to 40 for firmware-level smoothing
                            piper.MotionCtrl_2(0x01, 0x00, 40, 0x00)
                            piper.EndPoseCtrl(x_sdk, y_sdk, z_sdk, rx_sdk, ry_sdk, rz_sdk)
                            print(f"d[{dx_mm:+.1f}, {dy_mm:+.1f}, {dz_mm:+.1f}]mm | SDK: X={x_sdk} Y={y_sdk} Z={z_sdk} RX={rx_sdk} RY={ry_sdk} RZ={rz_sdk}   ", end="\r")
                        except Exception as e:
                            if not error_printed:
                                print("\n--- SDK COMMAND ERROR ---")
                                print(f"Arguments: X:{x_sdk}, Y:{y_sdk}, Z:{z_sdk}, RX:{rx_sdk}, RY:{ry_sdk}, RZ:{rz_sdk}")
                                print(f"Error: {repr(e)}")
                                print("-------------------------\n")
                                error_printed = True
                                
                        # Visual Telemetry
                        cv2.putText(frame, "STATUS: ACTIVE", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Robot X:{sm_x:.0f}mm Y:{sm_y:.0f}mm Z:{sm_z:.0f}mm", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, "Press 's' to PAUSE", (10, frame.shape[0] - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        cv2.putText(frame, "STATUS: WAITING FOR ORIGIN", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        cv2.putText(frame, "Press 'o' to SET ORIGIN", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    cv2.putText(frame, "STATUS: MARKER LOST", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Piper-X Relative Teleoperation", frame)
            
            # Throttle to ~100Hz (matching official demo's 0.01s sleep)
            elapsed = time.time() - start_time
            if elapsed < 0.01:
                time.sleep(0.01 - elapsed)

    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt, initiating shutdown...")
    except Exception as e:
        print(f"\nRuntime error encountered: {e}")
    finally:
        print("Stopping vision tracker...")
        tracker.stop()
        cv2.destroyAllWindows()
        
        print("Disabling Piper arm...")
        try:
            piper.DisablePiper()
        except Exception:
            try:
                piper.DisableArm(7)
            except:
                pass
            
        print("Teleoperation gracefully terminated.")

if __name__ == "__main__":
    main()
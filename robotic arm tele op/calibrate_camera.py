import cv2
import numpy as np

def main():
    # Configuration
    MARKER_ID = 0
    MARKER_LENGTH = 0.095  # 9.5 cm in meters
    CAMERA_INDEX = 1       # External webcam

    # Define the 3D coordinates of the marker corners in the marker's local frame
    # OpenCV convention: top-left, top-right, bottom-right, bottom-left
    l = MARKER_LENGTH / 2.0
    obj_points_single = np.array([
        [-l,  l, 0],
        [ l,  l, 0],
        [ l, -l, 0],
        [-l, -l, 0]
    ], dtype=np.float32)

    # ArUco dictionary (Assuming 4X4_50 as used in your tracker script)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}.")
        return

    # Try to force 640x480 resolution as used in the tracker
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    all_obj_points = []
    all_img_points = []
    image_size = None

    print("==================================================")
    print(" CAMERA CALIBRATION USING A SINGLE ARUCO MARKER")
    print("==================================================")
    print(f"Marker ID: {MARKER_ID}, Size: {MARKER_LENGTH * 100} cm")
    print("Instructions:")
    print(" - Move the marker around the camera view at different angles and distances.")
    print(" - Press 'c' to CAPTURE a frame.")
    print(" - Press 'k' to COMPUTE CALIBRATION (need at least 15-20 frames).")
    print(" - Press 'q' to QUIT.")
    print("==================================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        display_frame = frame.copy()
        
        marker_detected = False
        img_points = None
        
        if ids is not None:
            for i in range(len(ids)):
                if int(np.ravel(ids)[i]) == MARKER_ID:
                    marker_detected = True
                    cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
                    img_points = corners[i][0]
                    break

        # Display instructions and status on the frame
        cv2.putText(display_frame, f"Captured: {len(all_obj_points)} frames", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, "[C] Capture | [K] Calibrate | [Q] Quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Calibration", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Calibration aborted.")
            break
        elif key == ord('c'):
            if marker_detected:
                all_obj_points.append(obj_points_single)
                all_img_points.append(img_points)
                print(f"Captured frame {len(all_obj_points)}!")
            else:
                print("Marker not detected in this frame. Cannot capture.")
        elif key == ord('k'):
            if len(all_obj_points) < 10:
                print(f"Not enough frames! You only have {len(all_obj_points)}. Please capture at least 10 (ideally 20+).")
            else:
                print("\nCalibrating camera... This might take a second.")
                
                # We need to reshape the arrays slightly for calibrateCamera
                obj_points_list = [pts for pts in all_obj_points]
                img_points_list = [pts for pts in all_img_points]
                
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    obj_points_list, img_points_list, image_size, None, None
                )
                
                print(f"Calibration successful! RMS Error (pixels): {ret:.4f}")
                print("Camera Matrix (Intrinsics):")
                print(mtx)
                print("Distortion Coefficients:")
                print(dist)
                
                # Save to file
                output_file = "camera_calib.npz"
                np.savez(output_file, mtx=mtx, dist=dist)
                print(f"\nSaved calibration data to '{output_file}'.")
                print("The tracking script is now set up to automatically load this file!")
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


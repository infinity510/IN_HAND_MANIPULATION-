import cv2
import numpy as np
import time
import logging

class WebcamArucoTracker:
    """
    ArUco-based 3D marker tracking using a standard USB Webcam.
    Tracks Thumb (0), Index (1), and Middle (2) fingers using cv2.solvePnP.
    Provides Exponential Moving Average (EMA) filtering and robustness to temporary occlusion.
    """
    def __init__(self, marker_length=0.08, alpha=0.65, max_lost_frames=5, camera_index=0):
        """
        Args:
            marker_length (float): Physical edge length of the ArUco marker in meters (default: 0.015m = 15mm).
            alpha (float): EMA filter coefficient. Higher = more responsive, Lower = more smoothed.
            max_lost_frames (int): Number of frames to predict with constant velocity before freezing.
            camera_index (int): USB camera index. Usually 0 for laptop webcam, 1 or 2 for external USB webcams.
        """
        self.camera_index = camera_index
        
        # ArUco Configuration
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.marker_length = marker_length
        self.alpha = alpha
        self.max_lost_frames = max_lost_frames

        # 3D points of the marker corners in the marker's local coordinate system.
        l = self.marker_length / 2.0
        self.obj_points = np.array([
            [-l,  l, 0],
            [ l,  l, 0],
            [ l, -l, 0],
            [-l, -l, 0]
        ], dtype=np.float32)

        # State tracking for the 3 target markers: 0=Thumb, 1=Index, 2=Middle
        self.marker_ids = [0, 1, 2]
        
        # Initialize tracking state
        self.state = {
            m_id: {
                'pos_filtered': None,
                'vel': np.zeros(3),
                'lost_frames': 0,
                'last_time': None,
                'is_frozen': True
            } for m_id in self.marker_ids
        }
        
        self.cap = None
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))

    def start(self):
        """
        Starts the USB webcam pipeline and sets up approximate intrinsic calibration.
        """
        logging.info(f"Connecting to USB camera index {self.camera_index}...")
        # Initialize VideoCapture with V4L2 backend for Linux performance
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        
        # Try to force 640x480 resolution for consistency and performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # FIX LATENCY: Reduce buffer size to 1 so we always get the freshest frame, not a queued old one
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Give camera a moment to warm up
        time.sleep(1.0)
        
        if not self.cap.isOpened():
            raise Exception(f"Failed to open USB camera at index {self.camera_index}. Try changing camera_index!")

        # Since we don't have RealSense hardware intrinsics, we approximate them based on a standard 640x480 webcam.
        # This works perfectly fine for our teleoperation because any scaling error is absorbed by our POSITION_SCALING tuning factor in teleop_sim.py!
        fx = 600.0
        fy = 600.0
        cx = 320.0
        cy = 240.0
        
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        logging.info(f"WebcamArucoTracker started successfully on camera {self.camera_index}.")

    def stop(self):
        """
        Stops the USB webcam pipeline.
        """
        if self.cap is not None:
            self.cap.release()
        logging.info("WebcamArucoTracker stopped.")

    def step(self):
        """
        Fetches the next frame from webcam, detects markers, applies EMA + missing frame logic,
        and returns the estimated 3D Cartesian positions.
        """
        if self.cap is None or not self.cap.isOpened():
            return None, None

        ret, color_image = self.cap.read()
        if not ret:
            logging.warning("Failed to grab frame from USB camera!")
            return None, None

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, rejected = detector.detectMarkers(gray)

        detected_ids = set()
        current_time = time.time()

        if ids is not None:
            for i in range(len(ids)):
                m_id = int(np.ravel(ids)[i])
                if m_id not in self.marker_ids:
                    continue
                
                detected_ids.add(m_id)
                
                # Compute 3D pose of the marker relative to the camera
                success, rvec, tvec = cv2.solvePnP(
                    self.obj_points, corners[i][0], self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    pos_raw = tvec.flatten()
                    state = self.state[m_id]
                    
                    if state['pos_filtered'] is None or state['is_frozen']:
                        # Reset filter on first detection or recovery from frozen state
                        state['pos_filtered'] = pos_raw.copy()
                        state['vel'] = np.zeros(3)
                    else:
                        # EMA Filter for smooth trajectory
                        dt = current_time - state['last_time'] if state['last_time'] else 0.033
                        prev_pos = state['pos_filtered'].copy()
                        
                        state['pos_filtered'] = self.alpha * pos_raw + (1.0 - self.alpha) * prev_pos
                        
                        if dt > 0:
                            state['vel'] = (state['pos_filtered'] - prev_pos) / dt
                            
                    state['lost_frames'] = 0
                    state['last_time'] = current_time
                    state['is_frozen'] = False

        # Handle occlusion and missing markers
        for m_id in self.marker_ids:
            if m_id not in detected_ids:
                state = self.state[m_id]
                state['lost_frames'] += 1
                
                if state['pos_filtered'] is not None and not state['is_frozen']:
                    if state['lost_frames'] < self.max_lost_frames:
                        # Constant velocity prediction (dead reckoning) for short dropouts
                        dt = current_time - state['last_time'] if state['last_time'] else 0.033
                        state['pos_filtered'] += state['vel'] * dt
                        state['last_time'] = current_time
                    else:
                        # Safety: freeze if lost for too long
                        state['is_frozen'] = True
                        state['vel'] = np.zeros(3)

        # Construct the output dictionary
        tracking_out = {}
        for m_id in self.marker_ids:
            state = self.state[m_id]
            if state['pos_filtered'] is not None and not state['is_frozen']:
                tracking_out[m_id] = state['pos_filtered'].copy()
            else:
                tracking_out[m_id] = None
                
        # Optional: draw axes and markers for visual debugging
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            for i in range(len(ids)):
                m_id = int(np.ravel(ids)[i])
                if m_id in self.marker_ids:
                    success, rvec, tvec = cv2.solvePnP(
                        self.obj_points, corners[i][0], self.camera_matrix, self.dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    if success:
                        cv2.drawFrameAxes(color_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.01)

        return tracking_out, color_image

if __name__ == "__main__":
    # Quick visual validation test
    logging.basicConfig(level=logging.INFO)
    
    # Try index 4 which corresponds to the newly plugged in /dev/video4
    tracker = WebcamArucoTracker(marker_length=0.08, camera_index=4)
    tracker.start()
    
    try:
        while True:
            positions, frame = tracker.step()
            if frame is not None:
                # Print positions to console
                status = []
                for m_id in tracker.marker_ids:
                    pos = positions.get(m_id)
                    if pos is not None:
                        status.append(f"M{m_id}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                    else:
                        status.append(f"M{m_id}: LOST")
                print(" | ".join(status), end="\r")
                
                cv2.imshow("USB Webcam Aruco Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
        cv2.destroyAllWindows()

import cv2
import numpy as np
import pyrealsense2 as rs
import time
import logging

class ArucoTracker:
    """
    ArUco-based 3D marker tracking using Intel RealSense.
    Tracks Thumb (0), Index (1), and Middle (2) fingers using cv2.solvePnP.
    Provides Exponential Moving Average (EMA) filtering and robustness to temporary occlusion.
    """
    def __init__(self, marker_length=0.02, alpha=0.65, max_lost_frames=5):
        """
        Args:
            marker_length (float): Physical edge length of the ArUco marker in meters (default: 0.02m = 20mm).
            alpha (float): EMA filter coefficient. Higher = more responsive, Lower = more smoothed.
            max_lost_frames (int): Number of frames to predict with constant velocity before freezing.
        """
        # ArUco Configuration
        # Using 4x4 dictionary for smaller markers and faster detection
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.marker_length = marker_length
        self.alpha = alpha
        self.max_lost_frames = max_lost_frames

        # Intel RealSense Pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # 3D points of the marker corners in the marker's local coordinate system.
        # Order matches OpenCV's return order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
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
        
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))

    def start(self):
        """
        Starts the RealSense pipeline and retrieves the intrinsic calibration matrix.
        """
        profile = self.pipeline.start(self.config)
        
        # Retrieve camera intrinsics for precise metric 3D projection
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        # Build K matrix for solvePnP
        self.camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        logging.info("ArucoTracker started. Intrinsics retrieved.")

    def stop(self):
        """
        Stops the RealSense pipeline.
        """
        self.pipeline.stop()
        logging.info("ArucoTracker stopped.")

    def step(self):
        """
        Fetches the next frame, detects markers, applies EMA + missing frame logic,
        and returns the estimated 3D Cartesian positions.
        
        Returns:
            tracking_out (dict): Map of {marker_id: numpy array of shape (3,)} 
                                 Positions are None if the marker is frozen (lost).
            color_image (numpy array): The BGR frame from the camera, with optional debug drawing.
        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        detected_ids = set()
        current_time = time.time()

        if ids is not None:
            for i in range(len(ids)):
                m_id = ids[i][0]
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
                m_id = ids[i][0]
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
    tracker = ArucoTracker(marker_length=0.02)
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
                
                cv2.imshow("Aruco Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
        cv2.destroyAllWindows()

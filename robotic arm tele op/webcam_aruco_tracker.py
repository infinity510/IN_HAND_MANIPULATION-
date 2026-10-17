import cv2
import numpy as np
import time
import logging
import sys
import os
from collections import defaultdict
from scipy.spatial.transform import Rotation as R_scipy
import trimesh

class WebcamArucoTracker:
    """
    ArUco-based 3D object tracking using a standard USB Webcam.
    Tracks 5 ArUco markers (IDs 0-4) placed on a trapezoid-shaped gripper object (Gripper_updated.stl)
    and computes the full 6D object pose (position and 3x3 rotation matrix).
    
    Marker layout:
      - ID 0: Top face (PRIORITY marker, facing camera)
      - ID 1: Front side face (+X)
      - ID 2: Right side face (+Y)
      - ID 3: Back side face (-X)
      - ID 4: Left side face (-Y)
      
    Priority logic:
      - When ID 0 is visible, uses ONLY it for the pose.
      - When ID 0 is not visible, averages all visible side markers (IDs 1-4)
        and re-orthonormalizes the rotation matrix via SVD.
      - Skips frames where the rotation matrix is invalid.
    """
    def __init__(self, marker_length=0.06, alpha=0.65, max_lost_frames=5, camera_index=0, stl_path=None):
        """
        Args:
            marker_length (float): Physical edge length of the ArUco marker in meters (default: 0.06m = 60mm).
            alpha (float): EMA filter coefficient. Higher = more responsive, Lower = more smoothed.
            max_lost_frames (int): Number of frames to predict with constant velocity before freezing.
            camera_index (int): USB camera index. Usually 0 for laptop webcam, 1 or 2 for external USB webcams.
            stl_path (str, optional): Path to Gripper_updated.stl. If None, resolves relative to this file.
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

        # 5 Marker IDs
        self.marker_ids = [0, 1, 2, 3, 4]
        self.priority_id = 0
        
        # Camera to Robot base transformation matrix (from user's picture)
        self.R_cam2robot = np.array([
            [ 0, 0,  -1],
            [ 1,  0,  0],
            [ 0,  -1, 0]
        ], dtype=np.float64)

        # Resolve STL path
        if stl_path is None:
            this_dir = os.path.dirname(os.path.abspath(__file__))
            candidate_paths = [
                os.path.join(this_dir, "Gripper_updated.stl"),
                os.path.join(this_dir, "..", "Gripper_updated.stl"),
                os.path.join(os.getcwd(), "IN_HAND_MANIPULATION-", "robotic arm tele op", "Gripper_updated.stl"),
                os.path.join(os.getcwd(), "robotic arm tele op", "Gripper_updated.stl"),
                os.path.join(os.getcwd(), "Gripper_updated.stl")
            ]
            for p in candidate_paths:
                if os.path.exists(p):
                    stl_path = p
                    break
        
        self.stl_path = stl_path
        self._load_marker_layout_from_stl(self.stl_path)

        # State tracking for the fused object pose
        self.state = {
            'pos_filtered': None,
            'rotation_matrix': None,
            'euler_angles': None,
            'vel': np.zeros(3),
            'lost_frames': 0,
            'last_time': None,
            'is_frozen': True
        }
        
        self.cap = None
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        # Public properties for rotation outputs
        self.rotation_matrix = None
        self.euler_angles = None
        self.rotation_matrices = {m_id: None for m_id in self.marker_ids}
        self.origin_transform = None

    def _load_marker_layout_from_stl(self, stl_path):
        """
        Reads Gripper_updated.stl with trimesh to extract each marker face's normal and center.
        Computes marker offsets (in meters) and R_marker_to_object for IDs 0 to 4.
        """
        # Precomputed defaults (in meters) as fallback
        self.marker_offsets = {
            0: np.array([ 0.000000,  0.000000,  0.066668]),
            1: np.array([ 0.054445,  0.000000,  0.036112]),
            2: np.array([ 0.000000,  0.054445,  0.036112]),
            3: np.array([-0.054445,  0.000000,  0.036112]),
            4: np.array([ 0.000000, -0.054445,  0.036112]),
        }

        self.R_marker_to_object = {}
        
        mesh = None
        if stl_path and os.path.exists(stl_path):
            try:
                logging.info(f"Loading STL model from {stl_path}...")
                mesh = trimesh.load(stl_path)
            except Exception as e:
                logging.warning(f"Could not load STL with trimesh: {e}. Using precomputed layout.")
        else:
            logging.warning("STL path not found. Using precomputed marker layout.")

        if mesh is not None:
            # Cluster coplanar triangles by rounded normal
            normal_groups = defaultdict(list)
            for i, normal in enumerate(mesh.face_normals):
                key = tuple(np.round(normal, 3))
                normal_groups[key].append(i)

            groups = []
            for normal_key, tri_indices in normal_groups.items():
                areas = mesh.area_faces[tri_indices]
                total_area = areas.sum()
                centers = mesh.triangles_center[tri_indices]
                weighted_center = np.average(centers, weights=areas, axis=0)
                avg_normal = np.mean(mesh.face_normals[tri_indices], axis=0)
                avg_normal /= np.linalg.norm(avg_normal)
                groups.append({
                    'normal': avg_normal,
                    'center': weighted_center,
                    'area': total_area
                })

            groups.sort(key=lambda g: g['area'], reverse=True)
            
            # Top face (normal close to +Z)
            top_candidates = [g for g in groups if g['normal'][2] > 0.9]
            top_face = top_candidates[0] if top_candidates else None
            
            # 4 angled side faces
            side_faces = [g for g in groups if g['area'] > 5000 and (abs(g['normal'][0]) > 0.5 or abs(g['normal'][1]) > 0.5)]
            
            face_map = {}
            if top_face is not None:
                face_map[0] = top_face
                
            for s in side_faces:
                n = s['normal']
                if abs(n[0]) > abs(n[1]):
                    if n[0] > 0:
                        face_map[1] = s  # Front (+X)
                    else:
                        face_map[3] = s  # Back (-X)
                else:
                    if n[1] > 0:
                        face_map[2] = s  # Right (+Y)
                    else:
                        face_map[4] = s  # Left (-Y)

            # Update offsets from mesh if all 5 faces identified
            if len(face_map) == 5:
                for m_id, f_data in face_map.items():
                    self.marker_offsets[m_id] = f_data['center'] / 1000.0  # mm to meters

        # Compute R_marker_to_object for each ID
        # R_marker_to_object is constructed such that:
        # R_cam_object = R_cam_marker @ R_marker_to_object
        for m_id in self.marker_ids:
            if m_id == 0:
                # Top face: normal = [0, 0, 1]
                z_m = np.array([0.0, 0.0, 1.0])
                up_hint = np.array([1.0, 0.0, 0.0])
                x_m = np.cross(up_hint, z_m)
                x_m /= np.linalg.norm(x_m)
                y_m = np.cross(z_m, x_m)
                y_m /= np.linalg.norm(y_m)
            else:
                # Side face normal
                if m_id == 1:
                    normal = np.array([0.92847668, 0.0, -0.3713907])
                elif m_id == 2:
                    normal = np.array([0.0, 0.92847668, -0.3713907])
                elif m_id == 3:
                    normal = np.array([-0.92847668, 0.0, -0.3713907])
                elif m_id == 4:
                    normal = np.array([0.0, -0.92847668, -0.3713907])
                
                z_m = normal / np.linalg.norm(normal)
                up_hint = np.array([0.0, 0.0, 1.0])
                y_m = up_hint - np.dot(up_hint, z_m) * z_m
                y_m /= np.linalg.norm(y_m)
                x_m = np.cross(y_m, z_m)
                x_m /= np.linalg.norm(x_m)
                
            # R_marker_to_object has rows [x_m, y_m, z_m]
            self.R_marker_to_object[m_id] = np.vstack([x_m, y_m, z_m])

    @staticmethod
    def _is_valid_rotation(R, tol=0.1):
        """Checks if R is a valid rotation matrix (det close to 1, R^T R close to I)."""
        if R is None or R.shape != (3, 3):
            return False
        det = np.linalg.det(R)
        if abs(det - 1.0) > tol:
            return False
        orth_err = np.max(np.abs(R.T @ R - np.eye(3)))
        if orth_err > tol:
            return False
        return True

    @staticmethod
    def _orthonormalize_svd(R_mean):
        """Re-orthonormalizes an averaged rotation matrix using SVD ensuring det = +1."""
        U, S, Vt = np.linalg.svd(R_mean)
        R_ortho = U @ Vt
        if np.linalg.det(R_ortho) < 0:
            U[:, -1] *= -1
            R_ortho = U @ Vt
        return R_ortho

    def set_origin(self):
        """Sets the current pose of the object as the global origin."""
        state = self.state
        if state['pos_filtered'] is not None and not state['is_frozen']:
            T = np.eye(4)
            T[:3, :3] = state['rotation_matrix']
            T[:3, 3] = state['pos_filtered']
            self.origin_transform = T
            logging.info("Origin successfully set.")
            return True
        logging.warning("Failed to set origin: Object not visible.")
        return False

    def start(self):
        """
        Starts the USB webcam pipeline and sets up approximate intrinsic calibration.
        """
        logging.info(f"Connecting to USB camera index {self.camera_index}...")

        backend = cv2.CAP_ANY
        if sys.platform.startswith("linux"):
            backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
        elif sys.platform.startswith("win"):
            backend = getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)

        self.cap = cv2.VideoCapture(self.camera_index, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
        
        # Force 640x480 resolution for consistency and performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Buffer size 1 to minimize latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        time.sleep(1.0)
        
        if not self.cap.isOpened():
            raise Exception(f"Failed to open USB camera at index {self.camera_index}. Try changing camera_index!")

        # Check if a custom calibration file exists
        calib_file = os.path.join(os.path.dirname(__file__), "camera_calib.npz")
        if os.path.exists(calib_file):
            data = np.load(calib_file)
            self.camera_matrix = data['mtx']
            self.dist_coeffs = data['dist']
            logging.info("Loaded custom camera calibration from camera_calib.npz")
        else:
            fx = 600.0
            fy = 600.0
            cx = 320.0
            cy = 240.0
            self.camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float64)
            self.dist_coeffs = np.zeros((5, 1))
            logging.info("Using default approximate camera matrix.")
        
        logging.info(f"WebcamArucoTracker started successfully on camera {self.camera_index}.")

    def stop(self):
        """Stops the USB webcam pipeline."""
        if self.cap is not None:
            self.cap.release()
        logging.info("WebcamArucoTracker stopped.")

    def step(self):
        """
        Fetches next frame, detects up to 5 ArUco markers, computes object pose with priority logic,
        validates rotation matrix, applies EMA filter, and handles lost frames.
        
        Returns:
            tracking_out (dict): Fused pose under key 0 and 'object' (6D vector [x, y, z, rx, ry, rz]),
                                 plus per-marker poses if requested.
            color_image (np.ndarray): Annotated video frame.
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

        current_time = time.time()
        detected_marker_poses = {}
        marker_rvecs = {}
        marker_tvecs = {}

        if ids is not None:
            for i in range(len(ids)):
                m_id = int(np.ravel(ids)[i])
                if m_id not in self.marker_ids:
                    continue
                
                # Estimate 3D pose of this marker relative to camera
                success, rvec, tvec = cv2.solvePnP(
                    self.obj_points, corners[i][0], self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    marker_rvecs[m_id] = rvec
                    marker_tvecs[m_id] = tvec
                    
                    R_cam_marker, _ = cv2.Rodrigues(rvec)
                    
                    # Compute R_cam_object = R_cam_marker @ R_marker_to_object
                    R_cam_obj = R_cam_marker @ self.R_marker_to_object[m_id]
                    
                    # Compute object position from marker offset: pos_cam = tvec - R_cam_object @ offset
                    tvec_flat = tvec.flatten()
                    pos_cam_obj = tvec_flat - R_cam_obj @ self.marker_offsets[m_id]
                    
                    detected_marker_poses[m_id] = {
                        'pos': pos_cam_obj,
                        'R': R_cam_obj
                    }

        # Priority logic:
        # If top marker (ID 0) is visible, use ONLY it.
        # Otherwise, average visible side markers (IDs 1-4) and orthonormalize via SVD.
        fused_pos_cam = None
        fused_R_cam = None

        if self.priority_id in detected_marker_poses:
            fused_pos_cam = detected_marker_poses[self.priority_id]['pos']
            fused_R_cam = detected_marker_poses[self.priority_id]['R']
        else:
            side_visible = [m for m in [1, 2, 3, 4] if m in detected_marker_poses]
            if len(side_visible) > 0:
                pos_list = [detected_marker_poses[m]['pos'] for m in side_visible]
                R_list = [detected_marker_poses[m]['R'] for m in side_visible]
                
                fused_pos_cam = np.mean(pos_list, axis=0)
                if len(R_list) == 1:
                    fused_R_cam = R_list[0]
                else:
                    fused_R_cam = self._orthonormalize_svd(np.mean(R_list, axis=0))

        # Validate rotation matrix (skip frame if invalid)
        valid_detection = False
        if fused_R_cam is not None and fused_pos_cam is not None:
            if self._is_valid_rotation(fused_R_cam):
                valid_detection = True
            else:
                logging.warning("Invalid rotation matrix computed from markers. Skipping frame.")

        state = self.state

        if valid_detection:
            # Transform to robot base frame
            pos_raw = self.R_cam2robot @ fused_pos_cam
            rotation_matrix = self.R_cam2robot @ fused_R_cam
            
            # Double check rotation validity after robot transformation
            if self._is_valid_rotation(rotation_matrix):
                state['rotation_matrix'] = rotation_matrix
                state['euler_angles'] = R_scipy.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)
                
                if state['pos_filtered'] is None or state['is_frozen']:
                    state['pos_filtered'] = pos_raw.copy()
                    state['vel'] = np.zeros(3)
                else:
                    # EMA Filter
                    dt = current_time - state['last_time'] if state['last_time'] else 0.033
                    prev_pos = state['pos_filtered'].copy()
                    state['pos_filtered'] = self.alpha * pos_raw + (1.0 - self.alpha) * prev_pos
                    if dt > 0:
                        state['vel'] = (state['pos_filtered'] - prev_pos) / dt
                        
                state['lost_frames'] = 0
                state['last_time'] = current_time
                state['is_frozen'] = False
            else:
                valid_detection = False

        if not valid_detection:
            # Lost frame handling
            state['lost_frames'] += 1
            if state['pos_filtered'] is not None and not state['is_frozen']:
                if state['lost_frames'] < self.max_lost_frames:
                    # Constant velocity prediction
                    dt = current_time - state['last_time'] if state['last_time'] else 0.033
                    state['pos_filtered'] += state['vel'] * dt
                    state['last_time'] = current_time
                else:
                    state['is_frozen'] = True
                    state['vel'] = np.zeros(3)

        # Build output dictionary
        tracking_out = {}
        if state['pos_filtered'] is not None and not state['is_frozen']:
            if self.origin_transform is not None:
                T_curr = np.eye(4)
                T_curr[:3, :3] = state['rotation_matrix']
                T_curr[:3, 3] = state['pos_filtered']
                
                T_rel = np.linalg.inv(self.origin_transform) @ T_curr
                rel_pos = T_rel[:3, 3]
                rel_rot = T_rel[:3, :3]
                rel_euler = R_scipy.from_matrix(rel_rot).as_euler('xyz', degrees=False)
                pose_6d = np.concatenate([rel_pos, rel_euler])
            else:
                pose_6d = np.concatenate([state['pos_filtered'], state['euler_angles']])
            
            tracking_out['object'] = pose_6d
            # Keep key 0 backward compatible with existing teleop script
            tracking_out[0] = pose_6d
        else:
            tracking_out['object'] = None
            tracking_out[0] = None

        # Expose rotation matrix and euler angles
        self.rotation_matrix = (
            state['rotation_matrix'].copy() if state['rotation_matrix'] is not None else None
        )
        self.euler_angles = (
            state['euler_angles'].copy() if state['euler_angles'] is not None else None
        )
        self.rotation_matrices = {m_id: self.rotation_matrix for m_id in self.marker_ids}

        # Draw detected markers and axes for visual telemetry
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            for m_id in marker_rvecs:
                cv2.drawFrameAxes(color_image, self.camera_matrix, self.dist_coeffs,
                                  marker_rvecs[m_id], marker_tvecs[m_id], 0.04)

        return tracking_out, color_image

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tracker = WebcamArucoTracker(marker_length=0.06, camera_index=0)
    tracker.start()
    
    try:
        while True:
            positions, frame = tracker.step()
            if frame is not None:
                pose_6d = positions.get(0) if positions else None
                if pose_6d is not None:
                    print(f"Object: [{pose_6d[0]:.3f}, {pose_6d[1]:.3f}, {pose_6d[2]:.3f}, "
                          f"{pose_6d[3]:.3f}, {pose_6d[4]:.3f}, {pose_6d[5]:.3f}]", end="\r")
                else:
                    print("Object: LOST", end="\r")
                
                cv2.imshow("USB Webcam Aruco Tracking", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('o'):
                    if tracker.set_origin():
                        print("\n>>> ORIGIN SET! <<<\n")
                    else:
                        print("\n>>> Failed to set origin (object not visible). <<<\n")
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
        cv2.destroyAllWindows()

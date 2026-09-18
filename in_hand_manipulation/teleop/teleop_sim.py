import mujoco
import mujoco.viewer
import numpy as np
import time
import logging
import cv2
from pynput import keyboard
from dex_retargeting.retargeting_config import RetargetingConfig
from aruco_tracker import ArucoTracker
import os

logging.basicConfig(level=logging.INFO)

# --- Configuration ---
SCENE_XML = os.path.join(os.path.dirname(__file__), "../simulation/scene.xml")
RETARGET_YML = os.path.join(os.path.dirname(__file__), "delto_3f_retarget.yml")

CONTROL_HZ = 20
SIM_HZ = 500
SUBSTEPS = SIM_HZ // CONTROL_HZ
MAX_JOINT_VEL = 0.5  # rad per control step

# Map Camera coordinates to Robot coordinates
# Assumes overhead camera looking down at desk
R_CAM2ROB = np.array([
    [ 0, -1,  0],  # Cam Y (vertical) -> Robot -X (backward)
    [-1,  0,  0],  # Cam X (horizontal) -> Robot -Y (right)
    [ 0,  0, -1]   # Cam Z (depth) -> Robot -Z (down)
])
POSITION_SCALING = 2.5  # Amplified scaling so human pinches fully close the gripper


class TeleopSystem:
    """
    Teleoperation pipeline using Intel RealSense (ArUco markers) and dex-retargeting.
    Implements a clutch mechanism via the SPACEBAR.
    """
    def __init__(self):
        # 1. Load MuJoCo Model
        logging.info(f"Loading MuJoCo scene: {SCENE_XML}")
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        
        # 2. Setup Dex-Retargeting
        logging.info(f"Loading retargeting config: {RETARGET_YML}")
        self.retarget_config = RetargetingConfig.load_from_file(RETARGET_YML)
        self.retargeter = self.retarget_config.build()
        
        # 3. Identify link names and IDs for the 3 fingertips
        self.link_names = self.retarget_config.target_link_names
        self.link_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.link_names]
        
        # 4. Map target joints to position actuators
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                          for name in self.retarget_config.target_joint_names]
        self.actuator_ids = []
        for jid in self.joint_ids:
            found = False
            for i in range(self.model.nu):
                # If actuator drives a joint and the joint ID matches our target
                if self.model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT and self.model.actuator_trnid[i, 0] == jid:
                    self.actuator_ids.append(i)
                    found = True
                    break
            if not found:
                logging.warning(f"No position actuator found for joint ID {jid}")
                self.actuator_ids.append(-1)
                
        # 5. Initialize Tracker
        self.tracker = ArucoTracker(marker_length=0.015)
        
        # 6. Clutch State Variables
        self.clutch_active = False
        self.human_anchor = None  # Shape (3, 3)
        self.robot_anchor = None  # Shape (3, 3)
        self.last_qpos = None

    def on_press(self, key):
        if key == keyboard.Key.space and not self.clutch_active:
            self.clutch_active = True

    def on_release(self, key):
        if key == keyboard.Key.space:
            self.clutch_active = False

    def get_robot_tip_positions(self):
        """Returns shape (3, 3) array of current robot fingertip positions."""
        positions = []
        for body_id in self.link_ids:
            positions.append(self.data.xpos[body_id].copy())
        return np.array(positions)

    def run(self):
        self.tracker.start()
        
        # Start Keyboard Listener
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        
        # Reset and step simulation to initialize all kinematics
        mujoco.mj_resetData(self.model, self.data)
        
        # --- INITIALIZE TO 120 DEGREE TRIPOD GRASP ---
        # f1 is thumb (0 deg), f2 is index (-60 deg), f3 is middle (+60 deg)
        # This gives a symmetric starting pose so the IK solver can easily twist and sway!
        for name, val in [("gripper_f2m1_joint", -1.047), ("gripper_f3m1_joint", 1.047)]:
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid != -1:
                    self.data.qpos[self.model.jnt_qposadr[jid]] = val
            except: pass
            
        mujoco.mj_forward(self.model, self.data)
        
        # Initialize target joint positions
        qpos_indices = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.last_qpos = np.array([self.data.qpos[idx] for idx in qpos_indices])
        
        # VERY IMPORTANT: Tell dex-retargeting to start its optimizer from this tripod pose!
        if hasattr(self.retargeter, 'set_qpos'):
            self.retargeter.set_qpos(self.last_qpos)
            
        # Sync control array with initial positions to hold the gripper steady
        for i, aid in enumerate(self.actuator_ids):
            if aid != -1:
                self.data.ctrl[aid] = self.last_qpos[i]

        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                logging.info("=======================================")
                logging.info(" Teleoperation Sim Running! (20 Hz)    ")
                logging.info(" Hold SPACEBAR to clutch in and move.  ")
                logging.info(" Release SPACEBAR to pause teleop.     ")
                logging.info("=======================================")
                
                while viewer.is_running():
                    step_start = time.time()
                    
                    # 1. Fetch Tracking Data
                    tracking_out, frame = self.tracker.step()
                    
                    if frame is not None:
                        # Display clutch status on the camera feed
                        status_text = "CLUTCH: ACTIVE" if self.clutch_active else "CLUTCH: RELEASED"
                        color = (0, 255, 0) if self.clutch_active else (0, 0, 255)
                        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        cv2.imshow("ArUco Teleop Feed", frame)
                        cv2.waitKey(1)
                    
                    if tracking_out is not None:
                        # Extract 3D points for Thumb (0), Index (1), Middle (2)
                        valid_tracking = True
                        human_pos = []
                        for m_id in [0, 1, 2]:
                            pos = tracking_out.get(m_id)
                            if pos is None:
                                valid_tracking = False
                                break
                            human_pos.append(pos)
                        
                        # 2. Process Teleoperation if Clutched In
                        if self.clutch_active and valid_tracking:
                            human_pos = np.array(human_pos)
                            
                            # Clutch Just Engaged: set anchor frames
                            if self.human_anchor is None:
                                self.human_anchor = human_pos.copy()
                                self.robot_anchor = self.get_robot_tip_positions()
                                
                                # Tell Dex-Retargeting optimizer to start from the current robot pose!
                                if hasattr(self.retargeter, 'set_qpos'):
                                    self.retargeter.set_qpos(self.last_qpos)
                                    
                                logging.info("Clutched IN.")
                                
                            # Calculate Cartesian delta in camera frame
                            delta_human = human_pos - self.human_anchor
                            
                            # Transform to robot coordinate frame and scale
                            delta_robot = (delta_human @ R_CAM2ROB.T) * POSITION_SCALING
                            
                            # Compute desired absolute robot tip targets
                            target_robot_pos = self.robot_anchor + delta_robot
                            
                            # --- SAFETY BOUNDING BOX ---
                            # Prevent the IK solver from exploding if the ArUco markers jump or human moves too far!
                            # Workspace: X/Y within +/- 10cm, Z between +2cm and -15cm
                            target_robot_pos[:, 0] = np.clip(target_robot_pos[:, 0], -0.1, 0.1)
                            target_robot_pos[:, 1] = np.clip(target_robot_pos[:, 1], -0.1, 0.1)
                            target_robot_pos[:, 2] = np.clip(target_robot_pos[:, 2], -0.15, 0.02)
                            
                            # 3. Solve Inverse Kinematics
                            # retargeter.retarget() computes the optimized joint configuration
                            target_qpos = self.retargeter.retarget(target_robot_pos)
                            
                            # 4. Apply Rate Limiting (Safety Clamp)
                            delta_q = target_qpos - self.last_qpos
                            delta_q = np.clip(delta_q, -MAX_JOINT_VEL, MAX_JOINT_VEL)
                            safe_qpos = self.last_qpos + delta_q
                            
                            # Update command buffer
                            for i, aid in enumerate(self.actuator_ids):
                                if aid != -1:
                                    self.data.ctrl[aid] = safe_qpos[i]
                                    
                            self.last_qpos = safe_qpos
                            
                        elif not self.clutch_active and self.human_anchor is not None:
                            # Clutch Just Released: clear anchors
                            self.human_anchor = None
                            self.robot_anchor = None
                            logging.info("Clutched OUT.")
                            
                    # 5. Physics Integration
                    for _ in range(SUBSTEPS):
                        mujoco.mj_step(self.model, self.data)
                        
                    # Sync viewer UI
                    viewer.sync()
                    
                    # 6. Timing Lock (20 Hz Control Loop)
                    elapsed = time.time() - step_start
                    time_to_wait = (1.0 / CONTROL_HZ) - elapsed
                    if time_to_wait > 0:
                        time.sleep(time_to_wait)
                        
        except KeyboardInterrupt:
            logging.info("Shutting down...")
        finally:
            self.tracker.stop()
            listener.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    system = TeleopSystem()
    system.run()


import mujoco
import mujoco.viewer
import numpy as np
import time
import logging
import cv2
from pynput import keyboard
from dex_retargeting.retargeting_config import RetargetingConfig
from webcam_aruco_tracker import WebcamArucoTracker
import os

logging.basicConfig(level=logging.INFO)

SCENE_XML = os.path.join(os.path.dirname(__file__), "../simulation/scene.xml")
RETARGET_YML = os.path.join(os.path.dirname(__file__), "delto_3f_retarget.yml")

CONTROL_HZ = 20
SIM_HZ = 500
SUBSTEPS = SIM_HZ // CONTROL_HZ
MAX_JOINT_VEL = 0.5 

R_CAM2ROB = np.array([
    [ 0, -1,  0],  
    [-1,  0,  0],  
    [ 0,  0, -1]   
])
POSITION_SCALING = 1.5  

class TeleopSystem:
    def __init__(self):
        # Using the new USB Webcam tracker. 
        # Note: If your external USB webcam is not found, you may need to change camera_index=1 or 2
        self.tracker = WebcamArucoTracker(marker_length=0.015, camera_index=4)
        self.tracker.start()
        
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        
        config = RetargetingConfig.load_from_file(RETARGET_YML)
        self.retargeter = config.build()
        
        self.clutch_active = False
        self.absolute_mode = False
        self.trigger_reset = False
        self.human_anchor = None
        self.robot_anchor = None
        
        self.joint_names = self.retargeter.optimizer.robot.dof_joint_names
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        
        self.actuator_ids = []
        for name in self.joint_names:
            act_name = name.replace("gripper_", "act_").replace("_joint", "")
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            self.actuator_ids.append(act_id)

    def get_robot_tip_positions(self):
        # Use Dex-Retargeting's internal FK to prevent any anchor mismatch jumps!
        robot = self.retargeter.optimizer.robot
        robot.compute_forward_kinematics(self.last_qpos)
        positions = []
        for idx in self.retargeter.optimizer.target_link_indices:
            positions.append(robot.get_link_pose(idx)[:3, 3].copy())
        return np.array(positions)

    def on_press(self, key):
        if key == keyboard.Key.space:
            self.clutch_active = True
        try:
            if key.char == 'o':
                self.absolute_mode = not getattr(self, 'absolute_mode', False)
                logging.info(f"Absolute Mode (O-key): {self.absolute_mode}")
            elif key.char == 'r':
                self.trigger_reset = True
        except: pass

    def on_release(self, key):
        if key == keyboard.Key.space:
            self.clutch_active = False
            self.human_anchor = None
            self.robot_anchor = None

    def run(self):
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        
        mujoco.mj_resetData(self.model, self.data)
        
        # INITIALIZE TO A PINCHED 120-DEGREE TRIPOD GRASP
        initial_pose = [
            ("gripper_f1m1_joint", 0.0), ("gripper_f2m1_joint", -1.047), ("gripper_f3m1_joint", 1.047),
            ("gripper_f1m3_joint", 1.0), ("gripper_f1m4_joint", 1.0),
            ("gripper_f2m3_joint", 1.0), ("gripper_f2m4_joint", 1.0),
            ("gripper_f3m3_joint", 1.0), ("gripper_f3m4_joint", 1.0),
        ]
        for name, val in initial_pose:
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid != -1: self.data.qpos[self.model.jnt_qposadr[jid]] = val
            except: pass
            
        mujoco.mj_forward(self.model, self.data)
        qpos_indices = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.last_qpos = np.array([self.data.qpos[idx] for idx in qpos_indices])
        
        for i, aid in enumerate(self.actuator_ids):
            if aid != -1: self.data.ctrl[aid] = self.last_qpos[i]

        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                logging.info("=======================================")
                logging.info(" Teleop Running! (20 Hz)               ")
                logging.info(" Press 'o' to toggle ABSOLUTE MODE.    ")
                logging.info(" Hold SPACEBAR for relative mode.      ")
                logging.info("=======================================")
                
                while viewer.is_running():
                    step_start = time.time()
                    
                    if getattr(self, 'trigger_reset', False):
                        self.trigger_reset = False
                        
                        self.last_qpos = np.zeros_like(self.last_qpos)
                        reset_pose = [
                            ("gripper_f2m1_joint", -1.047), ("gripper_f3m1_joint", 1.047),
                            ("gripper_f1m3_joint", 1.0), ("gripper_f1m4_joint", 1.0),
                            ("gripper_f2m3_joint", 1.0), ("gripper_f2m4_joint", 1.0),
                            ("gripper_f3m3_joint", 1.0), ("gripper_f3m4_joint", 1.0),
                        ]
                        for name, val in reset_pose:
                            try:
                                idx = self.joint_names.index(name)
                                self.last_qpos[idx] = val
                            except: pass
                            
                        for i, aid in enumerate(self.actuator_ids):
                            if aid != -1: self.data.ctrl[aid] = self.last_qpos[i]
                            
                        for i, name in enumerate(self.joint_names):
                            try:
                                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                                if jid != -1:
                                    self.data.qpos[self.model.jnt_qposadr[jid]] = self.last_qpos[i]
                                    self.data.qvel[self.model.jnt_dofadr[jid]] = 0.0
                            except: pass
                            
                        mujoco.mj_forward(self.model, self.data)
                        if hasattr(self.retargeter, 'set_qpos'):
                            self.retargeter.set_qpos(self.last_qpos)
                            
                        self.clutch_active = False
                        self.human_anchor = None
                        self.robot_anchor = None
                        logging.info("Reset to initial starting pose.")
                        
                    tracking_out, frame = self.tracker.step()
                    
                    if frame is not None:
                        mode_text = "MODE: ABSOLUTE (O)" if self.absolute_mode else ("MODE: RELATIVE (SPACE)" if self.clutch_active else "MODE: PAUSED")
                        color = (0, 255, 255) if self.absolute_mode else ((0, 255, 0) if self.clutch_active else (0, 0, 255))
                        cv2.putText(frame, mode_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        cv2.imshow("ArUco Teleop Feed", frame)
                        cv2.waitKey(1)
                    
                    if tracking_out is not None:
                        valid_tracking = True
                        human_pos = []
                        for m_id in [0, 1, 2]:
                            pos = tracking_out.get(m_id)
                            if pos is None:
                                valid_tracking = False
                                break
                            human_pos.append(pos)
                        
                        target_robot_pos = None
                        direct_qpos = None
                        target_qpos = None
                        
                        # ==========================================
                        # ABSOLUTE MAPPING (Triggered by 'o')
                        # ==========================================
                        if self.absolute_mode and valid_tracking:
                            human_pos = np.array(human_pos)
                            centroid = np.mean(human_pos, axis=0)
                            centered = human_pos - centroid
                            
                            # 1. Find overall hand rotation (align Thumb to +X)
                            p0 = centered[0, :2]
                            p1 = centered[1, :2]
                            p2 = centered[2, :2]
                            v_human = p0 - (p1 + p2) / 2.0
                            v_len = np.linalg.norm(v_human) + 1e-6
                            v_human /= v_len
                            theta_hand = np.arctan2(v_human[1], v_human[0])
                            
                            # 2. Rotate all points into the canonical hand frame
                            c, s = np.cos(-theta_hand), np.sin(-theta_hand)
                            R_2d = np.array([[c, -s], [s, c]])
                            
                            aligned = np.zeros_like(centered)
                            for i in range(3):
                                aligned[i, :2] = R_2d @ centered[i, :2]
                                
                            # 3. Independent Finger Mapping (Curl Only)
                            # We lock the tripod (m1) and sway (m2) to guarantee the fingers NEVER cross or tangle!
                            curls = []
                            for i in range(3):
                                pt = aligned[i, :2]
                                r = np.linalg.norm(pt)
                                
                                # Distance from center -> Curl (m3, m4)
                                # Tuning: r=0.02 (pinch) -> curl=1.7 (touching), r=0.06 (open) -> curl=0.7 (flat)
                                curl_i = 1.7 - (r - 0.02) * 25.0
                                curl_i = np.clip(curl_i, -0.2, 2.0)
                                curls.append(curl_i)
                                
                            # 4. Apply to joints
                            direct_qpos = self.last_qpos.copy()
                            
                            # Lock m1 to perfect 120-degree tripod
                            for j_name, base_val in [("gripper_f1m1_joint", 0.0), ("gripper_f2m1_joint", -1.047), ("gripper_f3m1_joint", 1.047)]:
                                try:
                                    idx = self.joint_names.index(j_name)
                                    direct_qpos[idx] = base_val
                                except: pass
                                
                            # Apply independent curl (m3, m4) and lock sway (m2)
                            for i, f_idx in enumerate([1, 2, 3]):
                                # Lock Sway (m2) to 0
                                try:
                                    idx = self.joint_names.index(f"gripper_f{f_idx}m2_joint")
                                    direct_qpos[idx] = 0.0
                                except: pass
                                
                                # Apply Curl (m3, m4)
                                for m_idx in [3, 4]:
                                    try:
                                        idx = self.joint_names.index(f"gripper_f{f_idx}m{m_idx}_joint")
                                        direct_qpos[idx] = curls[i]
                                    except: pass

                        # ==========================================
                        # DECOUPLED INDEPENDENT TELEOP MAPPING
                        # ==========================================
                        elif self.clutch_active and valid_tracking and not self.absolute_mode:
                            human_pos = np.array(human_pos)
                            
                            # Calculate centroid to safeguard against global palm/arm movement!
                            current_centroid = np.mean(human_pos, axis=0)
                            centered = human_pos - current_centroid
                            
                            # Initialization at Clutch-In
                            if self.human_anchor is None:
                                # Anchor the CENTERED positions, not absolute world positions
                                self.human_anchor = {i: centered[i].copy() for i in range(3)}
                                self.init_robot_qpos = self.last_qpos.copy()
                                
                            target_qpos = self.init_robot_qpos.copy()
                            
                            # Keep base yaw joints (m1) locked in static tripod
                            for j_name, base_val in [("gripper_f1m1_joint", 0.0), ("gripper_f2m1_joint", -1.047), ("gripper_f3m1_joint", 1.047)]:
                                try:
                                    idx = self.joint_names.index(j_name)
                                    target_qpos[idx] = base_val
                                except: pass
                                
                            # Local spoke angles for Finger 1, 2, 3 (Matches the URDF physical mounting perfectly)
                            spoke_angles = [0.0, 2.094395, -2.094395]
                            
                            for i, f_idx in enumerate([1, 2, 3]):
                                # Delta is based on the CENTERED position. 
                                # If the palm moves, centered[] doesn't change, so delta is 0! (Immune to arm movement)
                                delta_human = centered[i] - self.human_anchor[i]
                                
                                # Noise Gate (Deadband filter)
                                if np.linalg.norm(delta_human) < 0.0015:
                                    delta_human = np.zeros(3)
                                    
                                delta_robot = (R_CAM2ROB @ delta_human) * POSITION_SCALING
                                dx, dy, dz = delta_robot
                                
                                # Project into local spoke frame
                                # Robot resting spoke angles (Matched to URDF)
                                spoke_angles = [0.0, 2.094395, -2.094395]
                                theta = spoke_angles[i]
                                c, s = np.cos(theta), np.sin(theta)
                                
                                delta_outward = dx * c + dy * s
                                delta_lateral = dx * (-s) + dy * c
                                delta_inward = -delta_outward
                                
                                # Joint Mapping Tuning (Increased sensitivity for faster response)
                                sway_delta = delta_lateral * 8.0  
                                curl_delta = delta_inward * 25.0 + dz * 15.0 
                                
                                # Apply lateral/horizontal displacements to Sway (m2)
                                try:
                                    idx = self.joint_names.index(f"gripper_f{f_idx}m2_joint")
                                    target_qpos[idx] = np.clip(self.init_robot_qpos[idx] + sway_delta, -0.6, 0.6)
                                except: pass
                                
                                # Apply vertical/inward displacements to Curl (m3, m4)
                                for m_idx in [3, 4]:
                                    try:
                                        idx = self.joint_names.index(f"gripper_f{f_idx}m{m_idx}_joint")
                                        target_qpos[idx] = np.clip(self.init_robot_qpos[idx] + curl_delta, -0.2, 2.2)
                                    except: pass
                                
                        elif not self.clutch_active:
                            self.human_anchor = None
                            
                        # ==========================================
                        # APPLY JOINTS (Bypassing IK Solver!)
                        # ==========================================
                        if direct_qpos is not None:
                            target_qpos = direct_qpos
                            
                        if target_qpos is not None:
                            delta_q = target_qpos - self.last_qpos
                            delta_q = np.clip(delta_q, -MAX_JOINT_VEL, MAX_JOINT_VEL)
                            safe_qpos = self.last_qpos + delta_q
                            
                            for i, aid in enumerate(self.actuator_ids):
                                if aid != -1:
                                    self.data.ctrl[aid] = safe_qpos[i]
                                    
                            self.last_qpos = safe_qpos
                            
                    for _ in range(SUBSTEPS):
                        mujoco.mj_step(self.model, self.data)
                        
                    viewer.sync()
                    
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

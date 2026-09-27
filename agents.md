# In-Hand Manipulation Project Documentation

## 1. Project Overview

This project focuses on **Dexterous In-Hand Manipulation**, specifically targeting the Tesollo Allegro-style robotic hand. The core objective is to collect human demonstrations via a low-cost, vision-based teleoperation setup (using ArUco markers and a webcam), and use this data to train a neural network policy.

**Core Pipeline:**
1. **Data Collection (MuJoCo):** A human operator teleoperates the simulated robot hand using a webcam tracking their physical fingers. The simulation records state-based trajectories of the hand and objects.
2. **Imitation Learning (Isaac Sim):** The collected `.hdf5` demonstrations are loaded into NVIDIA Isaac Sim to train a Behavioral Cloning policy.
3. **RL Fine-Tuning (Isaac Sim):** The IL policy is further refined using Reinforcement Learning in Isaac Sim.

**Key Features:**
* **Hindsight Experience Replay (HER) Labeling:** The data collection script automatically uses the final orientation of the object as the target goal for the episode, guaranteeing 100% success rate in the demonstration dataset.
* **Multi-Object Support:** Support for 5 different object shapes (Cylinder, Capsule, Cube, Cuboid, Sphere) to train generalized manipulation policies.
* **Zero-Lag Teleoperation:** Background daemon threading for OpenCV webcam capture to eliminate buffer latency.

---

## 2. Directory Structure

```
IN_HAND_MANIPULATION-main/
│
├── data/                                  # Recorded HDF5 demonstration trajectories
│
├── in_hand_manipulation/
│   ├── simulation/
│   │   ├── scene_cube.xml                 # MuJoCo scene with robot and all 5 objects
│   │   ├── scene.xml                      # MuJoCo scene (gripper only)
│   │   ├── tesollo_gym_env.py             # Gymnasium RL environment
│   │   └── ... (meshes/assets)
│   │
│   ├── teleop/
│   │   ├── collect_data.py                # Main data collection script (HER, Multi-object)
│   │   ├── teleop_sim.py                  # Sandbox teleop testing (no recording)
│   │   ├── webcam_aruco_tracker.py        # Threaded ArUco tracking pipeline
│   │   ├── aruco_tracker.py               # (Legacy) RealSense tracker
│   │   ├── test_rs.py                     # RealSense camera tester
│   │   └── dex_retargeting_config.yml     # Inverse Kinematics retargeting params
│
└── agents.md                              # This documentation file
```

---

## 3. Hardware & Teleoperation

The teleoperation system uses a standard 640x480 USB Webcam running at 60fps.
* **Markers:** 4 ArUco markers from `DICT_4X4_50` (15mm length).
    * `ID 0`: Thumb tip
    * `ID 1`: Index tip
    * `ID 2`: Middle tip
    * `ID 3`: Wrist / Palm (Used as the absolute origin to perfectly decouple finger movements).
* **Tracking (`webcam_aruco_tracker.py`):** Uses a background daemon thread (`_update_frame`) with a `threading.Lock` to continuously fetch the absolute latest frame from the webcam. This bypasses OpenCV's internal queue buffer, ensuring **zero-latency** state updates for the physics simulation.
* **Mapping and Calibration:**
    * The first time all 4 markers are detected (or when pressing `p`), the system captures a **relaxed grip** reference pose.
    * **Virtual Centroid (Curl):** The center of the 3 fingers relative to the wrist is fixed. Distance from this fixed centroid drives the curling joints (`m3`, `m4`), completely decoupling finger extension from global hand motion.
    * **Wrist Angle (Sway):** The side-to-side spread of each finger is calculated as the angle relative to the wrist marker (which is stable since the wrist is ~10cm away). This drives the base sway joint (`m2`).
    * **Clutch Mode:** (Hold Spacebar) Anchors the hand rotation, allowing twisting of the object using the base tripod joints (`m1`).

---

## 4. Multi-Object Data Collection Pipeline

The script `collect_data.py` is the heart of the project. It connects the webcam tracker to the MuJoCo physics engine and records HDF5 datasets explicitly formatted for downstream training in Isaac Sim.

### 4.1. Supported Objects
All objects have a mass of **1.0 kg** (defined in `scene_cube.xml`).
* `1`: Cylinder
* `2`: Cube
* `3`: Cuboid

### 4.2. Workflow
1. **Reset (`r`):** Resets the gripper to a tripod pinch and sweeps all objects out of the workspace.
2. **Spawn Object (`1` to `5`):** Teleports the selected object directly into the center of the gripper (`[0, 0, 0.1]`).
3. **Start Recording (`c`):** Begins saving the state of the robot and object at 50 Hz.
4. **Manipulate:** The operator freely rotates the object using their physical hand.
5. **Stop & Label (`c`):** Stops recording. The script looks at the final timestep, extracts the object's orientation, and saves it as the **Goal Orientation** (`target_quat`) for the entire episode.

---

## 5. Dataset Structure (HDF5)

The recorded `data/<object_name>_<index>.hdf5` files are heavily decoupled from MuJoCo to ensure seamless **Sim-to-Sim transfer to Isaac Sim (PhysX)**.

**Time-Series Datasets (Arrays of length T):**
* `robot_qpos` `(T, 12)`: Angular positions of the 12 gripper joints.
* `robot_qvel` `(T, 12)`: Angular velocities of the 12 gripper joints.
* `object_pos` `(T, 3)`: The `[X, Y, Z]` position of the active object.
* `object_quat` `(T, 4)`: The `[W, X, Y, Z]` rotation quaternion of the active object.
* `action` `(T, 12)`: The PD position targets sent to the motors. (Imitation Learning target).

**Episode Metadata (Attributes):**
* `f.attrs['object_id']`: Integer (1-5) representing the shape manipulated.
* `f.attrs['target_quat']`: The hindsight-labeled Goal Orientation `[W, X, Y, Z]`.

*(Note: When loading into Isaac Sim, ensure the quaternion format matches your specific library, e.g., PyTorch3D expects `[X, Y, Z, W]`).*

---

## 6. Training Pipeline (Isaac Sim)

The ultimate goal is to train a policy network in NVIDIA Isaac Sim.
1. **Data Preprocessing:** Before training, run `preprocess_data.py` to trim out static "dead time" at the beginning and end of recordings, ensuring the network doesn't learn to just stand still. This saves cleaned data to `data_processed/`.
2. **Behavioral Cloning (IL):** Write a PyTorch Dataloader to read the HDF5 files. Train a network to predict `action` given `(robot_qpos, robot_qvel, object_pos, object_quat, target_quat, object_id_one_hot)`. 
*(Note: `object_id_one_hot` is a 5-dimensional one-hot encoded vector representing the `object_id` from the episode metadata, ensuring the policy generalizes across different shapes.)*
3. **RL Fine-Tuning:** Deploy the pre-trained BC policy into an Isaac Sim Articulation environment. Use Reinforcement Learning (e.g., PPO via OmniIsaacGymEnvs) to fine-tune the policy to achieve the `target_quat` robustly, rewarding minimal distance to the goal orientation.

---

## 7. How to Run

### Data Collection (MuJoCo)
```bash
cd in_hand_manipulation/teleop
python collect_data.py
# Controls: 
# 'r' = Reset hand and clear workspace
# '1'-'3' = Spawn specific object shape (Cylinder, Cube, Cuboid)
# 'c' = Start/Stop recording (Hindsight goal labeling applies on stop)
# 'd' = Delete the latest saved recording
# 'p' = Recalibrate relaxed finger mapping
# SPACE = Clutch (engage/disengage human anchor for twisting)
# 'o' = Absolute mode toggle
```

### Replay Data
```bash
cd in_hand_manipulation/teleop
python replay_data.py
# Controls:
# SPACE = Pause / Play
# Left / Right Arrows = Skip to prev/next recording
```

### Data Preprocessing (For Isaac Sim)
```bash
cd in_hand_manipulation/teleop
python preprocess_data.py
# Reads from ../../data/ and outputs trimmed episodes to ../../data_processed/
```

### Sandbox Teleoperation (No Data Saved)
```bash
cd in_hand_manipulation/teleop
python teleop_sim.py
```

### Webcam Tracking Debugger
```bash
cd in_hand_manipulation/teleop
python webcam_aruco_tracker.py
```

---

## 8. Dependencies

| Package | Purpose |
|---------|---------|
| `mujoco` | Physics simulation engine for data collection |
| `h5py` | Recording and reading demonstration trajectories |
| `opencv-python` | Camera capture, ArUco detection, visual HUD |
| `numpy` / `scipy` | Numerical computation, Quaternion/Euler conversions |
| `pynput` | Asynchronous keyboard listener for teleop controls |


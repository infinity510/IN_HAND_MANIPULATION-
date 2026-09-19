# IN_HAND_MANIPULATION — Complete Project Context

> **Repository:** [github.com/infinity510/IN_HAND_MANIPULATION-](https://github.com/infinity510/IN_HAND_MANIPULATION-.git)  
> **Branches:** `main` (active), `lakshaya_teleop`, `agents/why-am-ib-gettin-g-this`

---

## 1. Project Overview

This project implements **dexterous in-hand manipulation** of objects using a **Tesollo / Delto DG-3F 3-finger robotic gripper**. The system combines:

1. **ArUco marker-based finger tracking** (via webcam or Intel RealSense depth camera) to teleoperate a physics-simulated gripper in **MuJoCo**
2. **Demonstration data collection** in HDF5 format for future **imitation learning / reinforcement learning**
3. A **Gymnasium RL environment** (`TesolloInHandEnv`) for training dexterous manipulation policies
4. A **robotic arm teleoperation** variant for Piper-X arm control using 6-DOF marker-based pose tracking

### High-Level Architecture

```
┌─────────────────────────┐      ┌────────────────────────────┐      ┌─────────────────────────────┐
│  Human Fingers with     │      │   ArUco Tracker             │      │  MuJoCo Simulation          │
│  ArUco Markers          │─────▶│  (Webcam / RealSense)       │─────▶│  (Delto-3F Gripper +        │
│  (IDs 0, 1, 2)          │      │  Detects markers,           │      │   Cube Object)              │
│                         │      │  computes tripod geometry    │      │                             │
│                         │      │  OR 6-DOF end-effector pose  │      │  12 joint position          │
│                         │      │                              │      │  actuators (4 per finger)   │
└─────────────────────────┘      └────────────────────────────┘      └──────────┬──────────────────┘
                                                                                │
                                                                                ▼
                                                                   ┌──────────────────────────┐
                                                                   │  HDF5 Data Files          │
                                                                   │  (qpos, qvel, action,     │
                                                                   │   cube pose,              │
                                                                   │   contact forces)         │
                                                                   └──────────────────────────┘
```

---

## 2. Gripper Specification — Tesollo Delto DG-3F

| Property | Value |
|----------|-------|
| **Robot Name** | `delto_3f_gripper` (URDF) / `tesollo_in_hand_scene` (MuJoCo) |
| **Fingers** | 3 (symmetrically arranged at 120° intervals on palm) |
| **DOF per finger** | 4 revolute: `m1` (yaw), `m2` (base pitch), `m3` (mid knuckle), `m4` (distal knuckle) |
| **Total actuated joints** | 12 |
| **Actuator type** | Position servo (`kp=15`, `kv=1`) |
| **Effort limit** | ±10 Nm per joint |
| **Velocity limit** | 5.0 rad/s (URDF) |
| **Total mass** | ~1 kg (base 0.370 kg + 3 × 0.208 kg per finger chain) |
| **Mesh files** | 6 STL: `base_link`, `link_01`–`link_04`, `link_tip_high` |
| **Model formats** | MuJoCo XML (`gripper_only.xml`) + ROS URDF (`gripper_only.urdf`) |

### Finger Placement on Palm
| Finger | Position (xyz) | Rotation (rpy) | m1 Yaw Range |
|--------|---------------|----------------|--------------|
| F1 (Thumb/Opposing) | `(0.0265, 0, 0)` | `(0, 0, 0)` | ±60° |
| F2 | `(-0.01334, 0.023, 0)` | `(0, 0, -π)` | -110° to +8° |
| F3 | `(-0.01334, -0.023, 0)` | `(0, 0, -π)` | -5° to +115° |

### Common Joint Limits (all fingers)
| Joint | Range |
|-------|-------|
| m2 (flexion) | ±101° |
| m3 (flexion) | -9° to +145° |
| m4 (flexion) | -13° to +116° |

---

## 3. Core Concept: Tripod Geometry Teleoperation

Three ArUco markers (IDs 0, 1, 2) are attached to the human's fingertips. The tracker computes:

| Metric       | Description                                                   | Maps To             |
|--------------|---------------------------------------------------------------|----------------------|
| **Spread**   | Mean distance of markers from their centroid                  | Grip open/close (m3/m4 curl) |
| **Twist**    | In-plane rotation angle of marker 0 around centroid           | Base yaw (m1) differential |
| **Sway**     | Tilt of the tripod plane (normal · vertical)                  | Lateral sway (m2) differential |
| **Centroid** | Mean 3D position of the three markers                         | Hand reference frame origin |
| **Normal**   | Surface normal of the plane formed by three markers           | Sway computation |

### Teleoperation Modes (`teleop_sim.py` / `collect_data.py`)

1. **Absolute Mode** (toggle `'o'` key):
   - Locks finger base yaw (m1) into symmetric 120° tripod: `[0°, -60°, +60°]`
   - Maps radial distance from centroid → proximal/distal curl (m3, m4)
   - Guarantees fingers never cross or tangle

2. **Relative / Clutch Mode** (hold `SPACEBAR`):
   - Captures anchor pose on clutch-in
   - Hand twist → coordinated base yaw rotation
   - Finger angle deviations → lateral sway (m2)
   - Finger spread → curling (m3, m4)
   - Release spacebar → freeze and disengage

3. **Safety**:
   - Joint velocity slew rate limiter: `MAX_JOINT_VEL = 0.5` rad/step
   - Dual-rate: 20 Hz control loop + 500 Hz physics (25 substeps)

---

## 4. Directory Structure

```
DC_PROJECT/
├── agents.md                              ← THIS FILE
├── .gitignore                             ← VS/Python gitignore
├── .gitmodules                            ← Submodule: github.com/infinity510/IN_HAND_MANIPULATION-
│
├── in_hand_manipulation/
│   ├── simulation/                        ← MuJoCo physics simulation
│   │   ├── tesollo_gym_env.py             ← Gymnasium RL environment (TesolloInHandEnv)
│   │   ├── scene.xml                      ← Base scene (gripper + commented-out cylinder)
│   │   ├── scene_cube.xml                 ← Self-contained scene with manipulable cuboid
│   │   ├── gripper_only.xml               ← Modular gripper kinematic tree (MuJoCo MJCF)
│   │   ├── gripper_only.urdf              ← Gripper model (ROS URDF, source of truth)
│   │   └── meshes/                        ← STL mesh files
│   │       ├── base_link.stl
│   │       ├── link_01.stl … link_04.stl
│   │       └── link_tip_high.stl
│   │
│   ├── teleop/                            ← Teleoperation pipeline (gripper)
│   │   ├── aruco_tracker.py               ← RealSense-based ArUco tracker (3 markers)
│   │   ├── webcam_aruco_tracker.py        ← Webcam-based ArUco tracker (3 markers)
│   │   ├── teleop_sim.py                  ← Main teleop: tracker → MuJoCo (Absolute + Clutch modes)
│   │   ├── collect_data.py                ← Data collection with cube, contacts, HDF5 recording
│   │   ├── test_rs.py                     ← RealSense hardware diagnostic
│   │   └── delto_3f_retarget.yml          ← Dex-retargeting config (IK retargeting)
│   │
│   └── robotic arm tele op/               ← Robotic arm variant (Piper-X)
│       ├── calibrate_camera.py            ← ChArUco camera calibration → camera_calib.npz
│       └── webcam_aruco_tracker.py        ← 6-DOF pose tracker (1 marker, camera→robot transform)
│
├── data/                                  ← Recorded HDF5 demonstration episodes
│
├── resources/                             ← Reference papers (PDFs)
│   ├── 1808.00177v5.pdf
│   ├── 2210.04887v1.pdf
│   ├── Dexterous_Manipulation_with_Deep_RL.pdf
│   └── p49.pdf
│
└── .vscode/                               ← VS Code settings
```

---

## 5. Detailed File Reference

### 5.1 Simulation Layer

---

#### `simulation/tesollo_gym_env.py` — Gymnasium RL Environment

**Class: `TesolloInHandEnv(gym.Env)`**

A full Gymnasium-compliant RL environment for training in-hand manipulation policies. The task: hold an object and reorient it to a target quaternion without dropping it.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(model_path="scene.xml", render_mode=None)` | Loads MuJoCo model, defines spaces, sets goal quaternion `[1,0,0,0]` |
| `_get_obs` | `()` | Returns 35-dim vector: `[qpos(12), qvel(12), obj_pos(3), obj_quat(4), goal_quat(4)]` |
| `reset` | `(seed, options)` | Resets simulation, returns `(obs, {})` |
| `step` | `(action)` | Rescales `[-1,1]` action to ctrl range, 10 substeps, returns `(obs, reward, terminated, truncated, {})` |
| `_compute_reward_and_status` | `()` | Dense reward: `-1.0 * orient_error - 2.0 * dist_to_palm`; terminates if `dist > 0.12m` |
| `render` | `()` | Launches/syncs passive MuJoCo viewer |
| `close` | `()` | Closes viewer |

**Spaces:**
- **Action:** `Box(12, low=-1.0, high=1.0)` — normalized position targets, rescaled to actuator ctrl ranges
- **Observation:** `Box(35)` — `[qpos(12), qvel(12), obj_pos(3), obj_quat(4), goal_quat(4)]`

**Reward:** Dense shaped reward combining orientation error (quaternion geodesic distance) and palm distance:
$$\text{reward} = -1.0 \times \underbrace{2 \arccos(|\mathbf{q}_{obj} \cdot \mathbf{q}_{goal}|)}_{\text{orient\_error}} - 2.0 \times \underbrace{\|\mathbf{p}_{obj} - \mathbf{p}_{palm}\|_2}_{\text{dist\_to\_palm}}$$

**Termination:** Object drops if `dist_to_palm > 0.12m`

**Physics:** 0.002s timestep × 10 substeps = 50 Hz control frequency

> [!NOTE]
> `_get_obs()` expects a body named `"target_object"`. In `scene.xml` this body is commented out; in `scene_cube.xml` it's named `"cube"`. Ensure the body name matches when switching scenes.

**Dependencies:** `gymnasium`, `mujoco`, `numpy`

---

#### `simulation/scene.xml` — Base MuJoCo Scene

Top-level scene that imports `gripper_only.xml` via `<include>`. Defines environment aesthetics and physics.

| Parameter | Value |
|-----------|-------|
| Timestep | `0.002s` (500 Hz) |
| Gravity | `0 0 -9.81` |
| Skybox | Blue-black gradient |
| Floor | Checkerboard grid, reflectance 0.2 |
| Light | Overhead directional |
| Target Object | **Commented out** — red cylinder (r=2.5cm, h=8cm, 100g) at `(0, 0, 0.25)` |

---

#### `simulation/scene_cube.xml` — Scene with Manipulable Object

Self-contained monolithic scene (does NOT use `<include>`). Inlines the full gripper definition plus a manipulable cuboid.

**Cube Object:**
| Property | Value |
|----------|-------|
| Name | `cube` |
| Joint | `cube_joint` (6-DOF freejoint) |
| Shape | Box: `0.035 × 0.035 × 0.07m` (7 × 7 × 14 cm) |
| Mass | `0.2 kg` |
| Initial position | `(0, 0, 0.1)` |
| Friction | `1 0.005 0.0001` |
| Pointer | Blue cylinder on top indicating heading direction |

**Collision Tuning:**
- Fingertip collision spheres: radius `12mm`, `contype=1 conaffinity=0`
- Cube geoms: `contype=0 conaffinity=1`
- This pairing enables fingertip↔cube contacts while eliminating self-collisions
- Solver: `solimp="0.99 0.99 0.001"`, `solref="0.002 1"` (stiff, stable)

---

#### `simulation/gripper_only.xml` — Modular Gripper (MuJoCo MJCF)

Isolates the robot definition for `<include>` embedding. 12 position actuators, 4-DOF × 3 finger kinematic chains. Fingertip collision uses actual STL meshes (no spheres, unlike `scene_cube.xml`).

---

#### `simulation/gripper_only.urdf` — Gripper (ROS URDF)

The **canonical source model** compiled from ROS `xacro`. Defines full inertial properties, visual/collision geometries, and joint limits. Referenced by `delto_3f_retarget.yml` for IK retargeting. Includes fixed tip links (`gripper_f{1,2,3}_tip_link`) that are merged into link_04 in MuJoCo conversion.

---

### 5.2 Teleoperation Layer

---

#### `teleop/aruco_tracker.py` — RealSense ArUco Tracker

**Class: `ArucoTracker`**

Tracks 3 ArUco markers (IDs 0, 1, 2) using an **Intel RealSense** depth camera with factory-calibrated intrinsics.

| Method | Description |
|--------|-------------|
| `__init__(marker_length=0.015, alpha=0.65, max_lost_frames=5)` | Configures RealSense 640×480@30fps, ArUco `DICT_4X4_50` |
| `start()` | Hardware resets RealSense (fixes "Frame didn't arrive" bug), starts pipeline, queries intrinsics |
| `stop()` | Stops RealSense pipeline |
| `step()` → `(tracking_out, color_image)` | Detects markers, `solvePnP`, EMA smoothing, dead-reckoning for occlusions |

**Key Features:**
- **EMA smoothing:** $p_{filtered} = \alpha \cdot p_{raw} + (1-\alpha) \cdot p_{prev}$, `alpha=0.65`
- **Dead reckoning:** On marker loss, extrapolates position via constant velocity for up to `max_lost_frames=5` frames, then freezes
- **Hardware reset** on startup to prevent USB pipeline lockup

**Dependencies:** `pyrealsense2`, `cv2`, `numpy`, `time`, `logging`

---

#### `teleop/webcam_aruco_tracker.py` — Webcam ArUco Tracker

**Class: `WebcamArucoTracker`** — Drop-in replacement for `ArucoTracker` using standard USB webcam.

| Parameter | RealSense (`ArucoTracker`) | Webcam (`WebcamArucoTracker`) |
|-----------|---------------------------|-------------------------------|
| Camera | Intel RealSense D435/D415 | Standard USB webcam |
| Intrinsics | Factory-calibrated from device | Synthetic: `fx=fy=600`, `cx=320`, `cy=240` |
| FPS target | 30 | 60 |
| `alpha` | `0.65` | `0.65` |
| Buffer size | N/A (RealSense SDK) | `1` (critical for zero-lag) |
| Backend | pyrealsense2 | `V4L2` (Linux) / `DSHOW` (Windows) |
| Dead reckoning | Yes | Yes |

Same API: `start()`, `stop()`, `step() → (tracking_out, color_image)`

**Dependencies:** `cv2`, `numpy`, `time`, `logging`, `sys`

---

#### `teleop/teleop_sim.py` — Main Teleoperation Script

**Class: `TeleopSystem`**

Bridges `WebcamArucoTracker` to MuJoCo simulation with two operating modes (Absolute + Clutch).

| Method | Description |
|--------|-------------|
| `__init__()` | Inits tracker (`camera_index=4`), loads `gripper_only.xml`, loads `RetargetingConfig` from YAML |
| `get_robot_tip_positions()` | Runs FK on retargeter to get 3D fingertip positions |
| `on_press/on_release(key)` | Handles SPACE (clutch), 'o' (absolute), 'r' (reset) |
| `run()` | Main 20 Hz loop: track → map → slew-limit → ctrl → 25 substeps → render |

**Mapping Pipeline (Absolute Mode):**
1. Compute hand centroid: $\bar{p} = \frac{1}{3}\sum p_i$
2. Compute hand orientation: $\theta_{hand} = \text{atan2}(v_y, v_x)$ where $\vec{v} = p_0' - \frac{p_1' + p_2'}{2}$
3. Canonical alignment: $aligned_i = R_{2D}(-\theta_{hand}) \cdot p_i'$
4. Curl mapping: $curl_i = 1.7 - (\|aligned_i\| - 0.02) \times 25.0 \in [-0.2, 2.0]$
5. Tripod base angles: `f1m1=0°, f2m1=-60°, f3m1=+60°`

**Constants:**
- `CONTROL_HZ = 20`, `SIM_HZ = 500`, `SUBSTEPS = 25`
- `MAX_JOINT_VEL = 0.5` rad/step

**Dependencies:** `mujoco`, `numpy`, `cv2`, `pynput.keyboard`, `dex_retargeting`, `webcam_aruco_tracker`, `time`, `logging`, `os`

---

#### `teleop/collect_data.py` — Data Collection with HDF5 Recording

**Class: `TeleopSystem`**

Extends the teleoperation system for demonstration recording.

**Additions over `teleop_sim.py`:**
- Loads `scene_cube.xml` (scene with manipulable cuboid)
- Press `'c'` to toggle recording on/off
- Records at ~50 Hz (downsampled: `Δt ≥ 0.02s`)
- **Contact sphere visualization**: Red spheres (`mjGEOM_SPHERE`, 5mm) at active MuJoCo contact positions
- **Cube orientation HUD**: Real-time roll/pitch/yaw via `scipy.spatial.transform.Rotation`
- Minimum 10 steps to save an episode

**HDF5 Output Schema (`data/episode_YYYYMMDD_HHMMSS.hdf5`):**

| Dataset | Shape per step | Type | Description |
|---------|---------------|------|-------------|
| `qpos` | `(n_qpos,)` | `float64` | Full simulation joint positions |
| `qvel` | `(n_qvel,)` | `float64` | Full simulation joint velocities |
| `action` | `(12,)` | `float64` | Applied actuator control signals (`ctrl`) |

**Cube Reset Pose:**
- Position: `(0, 0, 0.1)`, Quaternion: `(1, 0, 0, 0)` (identity)
- Initial gripper: `f1m1=0, f2m1=-1.047, f3m1=1.047`, curls `m3=m4=0.8`

**Dependencies:** `mujoco`, `numpy`, `cv2`, `pynput.keyboard`, `h5py`, `scipy`, `webcam_aruco_tracker`, `time`, `datetime`, `logging`, `os`

---

#### `teleop/test_rs.py` — RealSense Hardware Diagnostic

Standalone script: hardware resets RealSense, configures 640×480@30fps color stream, polls frames to verify connectivity. Useful for debugging "Frame didn't arrive within 5000" errors.

**Dependencies:** `pyrealsense2`, `time`

---

#### `teleop/delto_3f_retarget.yml` — Dex-Retargeting Configuration

YAML config for the `dex_retargeting` library providing IK retargeting from human fingertip positions to robot joint angles.

```yaml
retargeting:
  type: position                           # Position-based optimization (minimize Euclidean error)
  urdf_path: .../gripper_only.urdf         # Kinematics source
  target_joint_names:                      # 12 actuated joints
    - gripper_f1m1_joint ... gripper_f3m4_joint
  target_link_names:                       # 3 distal fingertip links
    - gripper_f1_04_link
    - gripper_f2_04_link
    - gripper_f3_04_link
  target_link_human_indices:               # Marker → Finger mapping
    - 0  →  Finger 1 (Thumb)
    - 1  →  Finger 2
    - 2  →  Finger 3
  has_joint_limits: true
  low_pass_alpha: 1.0                      # No internal filtering (deferred to tracker)
```

---

### 5.3 Robotic Arm Variant

---

#### `robotic arm tele op/calibrate_camera.py` — Camera Calibration

Interactive calibration utility using a single 9.5cm ArUco marker (ID 0, `DICT_4X4_50`).

**Workflow:**
1. Open webcam (device index **1**)
2. Press `'c'` to capture calibration frames (minimum 10, recommended 20+)
3. Press `'k'` to compute calibration via `cv2.calibrateCamera`
4. Saves `camera_matrix` + `dist_coeffs` → `camera_calib.npz`

**Dependencies:** `cv2`, `numpy`

---

#### `robotic arm tele op/webcam_aruco_tracker.py` — 6-DOF Pose Tracker (Piper-X Arm)

**Class: `WebcamArucoTracker`**

Specialized for **Piper-X robotic arm** teleoperation. Tracks a single marker (ID 0) and outputs full 6-DOF pose `[x, y, z, rx, ry, rz]`.

| Feature | Gripper `teleop/` version | Arm `robotic arm tele op/` version |
|---------|--------------------------|-------------------------------------|
| Markers tracked | 3 (IDs 0,1,2) | 1 (ID 0) |
| Output | 3D position per marker | 6D pose `[x,y,z,rx,ry,rz]` |
| Orientation | Not computed | Euler angles via `scipy.spatial.transform.Rotation` |
| Camera→Robot transform | None (raw camera coords) | `R_cam2robot` rotation matrix applied |
| Calibration | Synthetic intrinsics | Loads `camera_calib.npz` if available |
| Camera index | `0` or `4` | `1` |
| Dead reckoning | Yes | Yes (constant velocity extrapolation) |
| Marker length | 15mm | 15mm |

**Camera-to-Robot Transform:**
$$R_{cam2robot} = \begin{pmatrix} 0 & -1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & -1 \end{pmatrix}$$

**Dependencies:** `cv2`, `numpy`, `scipy`, `time`, `logging`, `sys`, `os`

---

## 6. Data Flow & Pipeline

```
                         TELEOPERATION LOOP
                         ═══════════════════

   ┌──────────┐   get_frame/step()  ┌──────────────┐   detect/solvePnP   ┌──────────┐
   │ Webcam / │───────────────────▶│  Color Frame │───────────────────▶│  Marker  │
   │ RealSense│                    │  (640×480)   │   + EMA filter     │  Poses   │
   └──────────┘                    └──────────────┘   + Dead Reckoning  └────┬─────┘
                                                                             │
                                                                 Absolute or Clutch Mode
                                                                 mapping equations
                                                                             │
                                                                             ▼
                                                                 ┌──────────────────┐
                                                                 │ 12 Joint Targets │
                                                                 │ (slew-limited)   │
                                                                 └────────┬─────────┘
                                                                          │
                                                                     data.ctrl[:12]
                                                                     × 25 substeps
                                                                          │
                                                                          ▼
                                                                 ┌──────────────────┐
                                                                 │ MuJoCo Physics   │
                                                                 │ + Passive Viewer │
                                                                 └────────┬─────────┘
                                                                          │
                                                               (if recording, @50Hz)
                                                                          │
                                                                          ▼
                                                                 ┌──────────────────┐
                                                                 │ HDF5 File        │
                                                                 │ data/episode_*.h5│
                                                                 └──────────────────┘
```

---

## 7. Inter-File Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │           UPSTREAM MODEL SOURCE              │
                    │                                              │
                    │  gripper_only.urdf (xacro-compiled)          │
                    │    - Canonical kinematics, inertias          │
                    │    - Referenced by delto_3f_retarget.yml     │
                    └───────────────┬──────────────────────────────┘
                                    │ converted to MJCF
                                    ▼
                    ┌───────────────────────────────┐
                    │     gripper_only.xml          │
                    │  (modular, for <include>)     │
                    └──────────┬────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │ <include>      │                │ loaded directly
              ▼                │                ▼
     ┌────────────────┐        │    ┌────────────────────────┐
     │   scene.xml    │        │    │   teleop_sim.py        │
     │  (gripper only,│        │    │  (bare gripper teleop) │
     │  no object)    │        │    └────────────────────────┘
     └───────┬────────┘        │
             │                 │
             ▼                 │
     ┌────────────────┐        │
     │tesollo_gym_env │        │    ┌────────────────────────┐
     │  (RL training) │        │    │   scene_cube.xml       │
     └────────────────┘        │    │  (self-contained,      │
                               │    │   gripper + cube)      │
                               │    └──────────┬─────────────┘
                               │               │
                               │               ▼
                               │    ┌────────────────────────┐
                               │    │   collect_data.py      │
                               │    │  (demo recording)      │
                               │    └────────────────────────┘
                               │
     ┌─────────────────────────┼──────────────────────┐
     │            TRACKER LAYER                       │
     │                                                │
     │  aruco_tracker.py (RealSense) ─── Alternative ─│── webcam_aruco_tracker.py (Webcam)
     │                                                │         │
     │                                                │         ▼
     │                                                │  teleop_sim.py & collect_data.py
     └────────────────────────────────────────────────┘

     ┌────────────────────────────────────────────────┐
     │        ROBOTIC ARM VARIANT                     │
     │                                                │
     │  calibrate_camera.py → camera_calib.npz        │
     │                              │                 │
     │                              ▼                 │
     │  webcam_aruco_tracker.py (6-DOF, R_cam2robot)  │
     │  → Piper-X arm teleoperation                   │
     └────────────────────────────────────────────────┘
```

---

## 8. Dependencies & Requirements

### Python Packages

| Package | Used By | Purpose |
|---------|---------|---------|
| `mujoco` | `tesollo_gym_env.py`, `teleop_sim.py`, `collect_data.py` | Physics simulation engine |
| `gymnasium` | `tesollo_gym_env.py` | RL environment interface |
| `opencv-python` (`cv2`) | All tracker/teleop scripts | Camera capture, ArUco detection, visualization |
| `numpy` | All Python files | Numerical computation |
| `pyrealsense2` | `aruco_tracker.py`, `test_rs.py` | Intel RealSense camera SDK |
| `h5py` | `collect_data.py` | HDF5 data recording |
| `scipy` | `collect_data.py`, `robotic arm tele op/webcam_aruco_tracker.py` | Quaternion → Euler angle conversion |
| `pynput` | `teleop_sim.py`, `collect_data.py` | Keyboard listener (clutch, mode toggle, record) |
| `dex_retargeting` | `teleop_sim.py` | IK-based retargeting from human to robot joints |

### Hardware

| Device | Required For | Notes |
|--------|-------------|-------|
| USB Webcam | `webcam_aruco_tracker.py`, `teleop_sim.py`, `collect_data.py` | 640×480 @60fps, buffer size 1 |
| Intel RealSense D435/D415 | `aruco_tracker.py`, `test_rs.py` | Depth camera (optional alternative) |
| ArUco Markers | All tracking | `DICT_4X4_50`, IDs 0–2 printed at 15mm |
| ChArUco Board / 9.5cm ArUco | `calibrate_camera.py` | For camera intrinsic calibration |

---

## 9. How to Run

### Teleoperation (Webcam → Gripper Simulation)
```bash
cd in_hand_manipulation/teleop
python teleop_sim.py
# Controls: SPACE=clutch, 'o'=absolute mode, 'r'=reset, 'q'=quit
```

### Data Collection (with Cube)
```bash
cd in_hand_manipulation/teleop
python collect_data.py
# Controls: 'c'=toggle recording, SPACE=clutch, 'o'=absolute, 'r'=reset
# Data saved to: data/episode_YYYYMMDD_HHMMSS.hdf5
```

### RL Environment Test
```bash
cd in_hand_manipulation/simulation
python tesollo_gym_env.py
# Runs 500 random action steps with visualization
```

### RealSense Camera Test
```bash
cd in_hand_manipulation/teleop
python test_rs.py
```

### Camera Calibration (Robotic Arm)
```bash
cd "in_hand_manipulation/robotic arm tele op"
python calibrate_camera.py
# Press 'c' to capture frames, 'k' to calibrate and save
```

---

## 10. Development History

The git history reveals iterative development (20 commits on `main`):

| Phase | Commits | Key Changes |
|-------|---------|-------------|
| **v1** | `90f2b03` | Initial implementation |
| **Webcam Support** | `bf0f17a` | Added webcam tracker as RealSense alternative |
| **Latency Optimization** | `a8577ff`, `8f8eb85` | Eliminated tracking latency, disabled EMA, 60fps webcam |
| **Data Collection** | `64be444`, `4e7f4ea` | HDF5 recording, contact visualization, 'c' key toggle, 50Hz downsample |
| **Physics Stability** | `69715bd`–`a6d7c4f` | Fixed exploding contact forces, collision groups |
| **Object Tuning** | `132923c`–`72c0fb2` | Iterative cube sizing (7cm → 8.4cm → doubled) |
| **Collision Hardening** | `8808a64`, `7f44c85` | Fingertip collision spheres, stiffened solver, loosened initial pose |
| **Visual & Physics** | `4135c1d`, `f5a778a` | Blue direction pointer, fixed 10× slow-motion physics substep bug |
| **Final Polish** | `8f8eb85` | Disabled EMA, relaxed velocity limits, 60fps webcam request |

---

## 11. Reference Papers

| File | Likely ArXiv ID / Topic |
|------|------------------------|
| `1808.00177v5.pdf` | Dexterous manipulation methods |
| `2210.04887v1.pdf` | In-hand manipulation / RL approaches |
| `Dexterous_Manipulation_with_Deep_RL.pdf` | Deep RL for dexterous manipulation (efficient, general, low-cost) |
| `p49.pdf` | Additional reference |

---

## 12. Future / Planned Work

Based on the codebase structure and existing components:

| Direction | Evidence |
|-----------|----------|
| **Imitation Learning** | HDF5 demonstration data being collected (`collect_data.py`) |
| **RL Policy Training** | Full Gymnasium environment with shaped reward (`TesolloInHandEnv`) |
| **Sim-to-Real Transfer** | URDF model + `dex_retargeting` config prepared for physical deployment |
| **Robotic Arm Integration** | `robotic arm tele op/` directory with 6-DOF tracker and Piper-X arm targeting |
| **Multi-camera support** | Camera index configurability (0, 1, 2, 4) suggests multi-device setups |


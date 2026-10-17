#!/usr/bin/env python3
"""
evaluate_dr.py
=============================================================================
Domain Randomization (Sim-to-Real) Benchmark Evaluator for Delto Gripper-3F.

This evaluator tests the policy under rigorous real-world perturbations:
  1. Actuation Latency Buffer: Random 0 to N-step transport delay (0 - 40 ms lag,
     matching CAN bus and real servo communication loops).
  2. Observation Sensor Noise: Joint encoder jitter (±0.012 rad), velocity noise
     (±0.04 rad/s), vision pose drift (±3 mm pos, ±0.02 rad orientation).
  3. Dynamic Silicone Friction: Variable fingertip and palm friction (μ in [2.1, 3.9],
     matching real silicone wear, degradation, and temperature shifts).
  4. Physical Property Variation: ±30% randomized mass, inertia, and dimensions.
  5. Non-Privileged Observations: Policy receives nominal (unprivileged) physical
     parameters rather than ground-truth simulator properties.

Stratified Angle Range Evaluation Protocol:
  - Range 1 (Small-to-Medium Rotations): 10.0° to 40.0° (0.1745 to 0.6981 rad)
  - Range 2 (Large-Angle Gaiting)     : 40.0° to 90.0° (0.6981 to 1.5708 rad)

Standard Evaluation Envelope:
  - 100 episodes per range per object type (or configurable via CLI)
  - 3 Object Types: Cylinder (obj_1), Cube (obj_2), Cuboid (obj_3)
  - Total: 100 × 2 ranges × 3 objects = 600 total episodes

Features:
  - Multi-worker parallel CPU execution (16 workers default).
  - Optional visual 3D MuJoCo rendering with live DR telemetry HUD.
  - Stratified metric reports for Range 1, Range 2, and Grand Composite.
  - Generates comprehensive markdown report saved to EVALUATION_DR_REPORT.md.
=============================================================================
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import scipy.spatial.transform as transform
import torch
import multiprocessing as mp
import mujoco

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from in_hand_manipulation.train_ppo import DeltoInHandManipulationEnv, ActorNetwork

OBJECT_NAMES = {1: "Cylinder", 2: "Cube", 3: "Cuboid"}


def _shuffle_avoid_adjacent_repeats(items):
    """Permutes episode items while attempting to avoid back-to-back identical objects."""
    items = list(items)
    for _ in range(50):
        random.shuffle(items)
        if not any(items[i][0] == items[i + 1][0] for i in range(len(items) - 1)):
            return items
    return items


RANGE_DEFINITIONS = [
    {
        "id": 1,
        "name": "Range 1 (10° to 40° Small/Medium)",
        "min_deg": 10.0,
        "max_deg": 40.0,
        "min_rad": np.radians(10.0),
        "max_rad": np.radians(40.0),
    },
    {
        "id": 2,
        "name": "Range 2 (40° to 90° Large Gaiting)",
        "min_deg": 40.0,
        "max_deg": 90.0,
        "min_rad": np.radians(40.0),
        "max_rad": np.radians(90.0),
    },
]

# Multiprocessing worker cache
_w_actor = None
_w_env = None
_w_epsilon_ratio = 0.10
_w_smooth_alpha = 1.0


def _worker_init(
    state_dict,
    obs_mean,
    obs_std,
    scene_xml_path,
    epsilon_ratio,
    smooth_alpha,
    action_delay_max,
    vision_delay_max,
    obs_noise_scale,
    dr_ratio,
):
    global _w_actor, _w_env, _w_epsilon_ratio, _w_smooth_alpha
    torch.set_num_threads(1)
    _w_actor = ActorNetwork(obs_dim=39, act_dim=12)
    _w_actor.load_state_dict(state_dict, strict=False)
    _w_actor.eval()
    _w_epsilon_ratio = epsilon_ratio
    _w_smooth_alpha = smooth_alpha
    _w_env = DeltoInHandManipulationEnv(
        scene_xml_path=scene_xml_path,
        obs_mean=obs_mean,
        obs_std=obs_std,
        max_episode_steps=250,
        max_goal_angle=1.5708,
        dr_ratio=dr_ratio,
        steps_per_radian=200,
        bonus_steps_per_success=150,
        max_steps_cap=800,
        sim2real_dr=True,
        action_delay_max=action_delay_max,
        vision_delay_max=vision_delay_max,
        smooth_alpha=smooth_alpha,
        obs_noise_scale=obs_noise_scale,
    )


def _eval_single_task(task_args):
    """Worker task execution function for headless multiprocessing pool under Domain Randomization."""
    global _w_actor, _w_env, _w_epsilon_ratio, _w_smooth_alpha
    seed, range_id, obj_id, min_rad, max_rad = task_args
    np.random.seed(seed)

    obs, _ = _w_env.reset(object_id=obj_id)

    # Sample uniform axis and uniform angle in specified range
    if obj_id == 1:
        # Cylinder: dominant longitudinal spin with bounded tilt/nutation
        axis = np.array([np.random.uniform(-0.25, 0.25), np.random.uniform(-0.25, 0.25), np.random.choice([-1.0, 1.0])])
        axis /= np.linalg.norm(axis)
    else:
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis) + 1e-8

    target_angle = float(np.random.uniform(min_rad, max_rad))
    rand_rot = transform.Rotation.from_rotvec(axis * target_angle)
    q_xyzw = rand_rot.as_quat()
    _w_env.target_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

    obj_idx = _w_env.model.jnt_qposadr[_w_env.model.joint(f"obj_{obj_id}_joint").id]
    cur_obj_quat = _w_env.data.qpos[obj_idx + 3 : obj_idx + 7]
    _w_env.prev_d_theta = _w_env.compute_angular_distance(cur_obj_quat, _w_env.target_quat)
    _w_env.initial_d_theta = float(_w_env.prev_d_theta)

    init_angle_deg = float(np.degrees(_w_env.initial_d_theta))
    theta_rad = float(_w_env.initial_d_theta)
    quad_steps = int(250 + 200 * theta_rad + 150.0 * (theta_rad**2))
    _w_env.current_max_steps = min(800, quad_steps)
    _w_env.set_epsilon_ratio(_w_epsilon_ratio)
    _w_env.set_epsilon(_w_epsilon_ratio * theta_rad)

    # Refresh observation with active DR noise and the new target_quat
    obs = _w_env._get_observation()

    step = 0
    terminated = False
    truncated = False
    step_info = {}
    peak_torque = 0.0

    while not (terminated or truncated):
        step += 1
        with torch.no_grad():
            raw_act = _w_actor(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).numpy()
        obs, reward, terminated, truncated, step_info = _w_env.step(raw_act)

        cur_torque = float(np.max(np.abs(_w_env.data.actuator_force)))
        if cur_torque > peak_torque:
            peak_torque = cur_torque

        goals = step_info.get("successes_in_episode", 0)
        if goals >= 1:
            break

    final_err_deg = step_info.get("d_theta", 0.0) * (180.0 / np.pi)
    raw_dropped = bool(step_info.get("raw_dropped", False))
    goals_achieved = step_info.get("successes_in_episode", 0)

    # Success criteria at epsilon tolerance
    tol_thresh_deg = _w_epsilon_ratio * init_angle_deg
    is_success = bool(goals_achieved > 0 or final_err_deg <= tol_thresh_deg)

    # Pre-goal retention accounting
    dropped_before_success = bool(raw_dropped and (goals_achieved == 0))
    pre_goal_retained = not dropped_before_success

    if goals_achieved > 0:
        closure = 1.0
    elif dropped_before_success:
        closure = 0.0
    else:
        closure = max(0.0, 1.0 - (final_err_deg / (init_angle_deg + 1e-6)))

    return {
        "range_id": range_id,
        "object_id": obj_id,
        "init_angle_deg": init_angle_deg,
        "final_err_deg": final_err_deg,
        "is_success": is_success,
        "closure": closure,
        "raw_dropped": raw_dropped,
        "pre_goal_retained": pre_goal_retained,
        "dropped_before_success": dropped_before_success,
        "goals_achieved": goals_achieved,
        "steps": step,
        "peak_torque": peak_torque,
        "action_delay_steps": getattr(_w_env, "active_action_delay", 0),
    }


def _add_user_sphere(scn, pos, radius, rgba):
    """Adds a decorative sphere to the MuJoCo viewer user scene."""
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, radius, radius], dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten().astype(np.float64),
        rgba=np.array(rgba, dtype=np.float32),
    )
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    scn.ngeom += 1


def _add_user_arrow(scn, p1, p2, radius, rgba):
    """Adds a decorative connector arrow to the MuJoCo viewer user scene."""
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3, dtype=np.float64),
        pos=np.zeros(3, dtype=np.float64),
        mat=np.eye(3).flatten().astype(np.float64),
        rgba=np.array(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        g,
        mujoco.mjtGeom.mjGEOM_ARROW,
        radius,
        np.asarray(p1, dtype=np.float64),
        np.asarray(p2, dtype=np.float64),
    )
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    scn.ngeom += 1


def run_rendered_evaluation(
    checkpoint_path: str,
    scene_xml_path: str,
    episodes_per_range: int,
    epsilon_ratio: float,
    action_delay_max: int = 1,
    vision_delay_max: int = 2,
    obs_noise_scale: float = 0.80,
    dr_ratio: float = 0.30,
    hold_time: float = 1.2,
    max_goals: int = 1,
    smooth_alpha: float = 0.50,
    sample_mode: str = "shuffle",
    report_file: str = "EVALUATION_DR_REPORT.md",
):
    """Interactive visual evaluation with MuJoCo passive 3D viewer with active DR."""
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as e:
        print(f"Error importing mujoco.viewer: {e}")
        return

    print(f"\n[Viewer] Launching Interactive MuJoCo Visual Domain Randomization Evaluation: {checkpoint_path}")
    print(f"  DR Settings: action_delay_max={action_delay_max} (up to {action_delay_max*20}ms) | "
          f"vision_delay_max={vision_delay_max} ({vision_delay_max*20}ms ArUco delay) | "
          f"obs_noise_scale={obs_noise_scale:.2f} | dr_ratio=±{int(dr_ratio*100)}%\n")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("actor_state_dict", payload.get("model_state_dict"))
    obs_mean = payload["obs_mean"]
    obs_std = payload["obs_std"]

    actor = ActorNetwork(obs_dim=39, act_dim=12)
    actor.load_state_dict(state_dict, strict=False)
    actor.eval()

    env = DeltoInHandManipulationEnv(
        scene_xml_path=scene_xml_path,
        obs_mean=obs_mean,
        obs_std=obs_std,
        max_episode_steps=250,
        max_goal_angle=1.5708,
        dr_ratio=dr_ratio,
        steps_per_radian=200,
        bonus_steps_per_success=150,
        max_steps_cap=800,
        sim2real_dr=True,
        action_delay_max=action_delay_max,
        vision_delay_max=vision_delay_max,
        smooth_alpha=smooth_alpha,
        obs_noise_scale=obs_noise_scale,
    )

    all_results = []
    print("\n" + "=" * 80)
    print("MUJOCO VISUAL VIEWER ACTIVE (DOMAIN RANDOMIZATION) - Press SPACE to pause, ESC to exit")
    print("=" * 80 + "\n")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for r_cfg in RANGE_DEFINITIONS:
            r_id = r_cfg["id"]
            r_name = r_cfg["name"]
            min_rad = r_cfg["min_rad"]
            max_rad = r_cfg["max_rad"]

            if sample_mode == "sequential":
                episodes = [(obj_id, ep) for obj_id in [1, 2, 3] for ep in range(1, episodes_per_range + 1)]
            elif sample_mode == "random":
                total_eps = episodes_per_range * 3
                episodes = [(int(np.random.choice([1, 2, 3])), ep) for ep in range(1, total_eps + 1)]
            else:
                ep_pool = [(obj_id, ep) for obj_id in [1, 2, 3] for ep in range(1, episodes_per_range + 1)]
                episodes = _shuffle_avoid_adjacent_repeats(ep_pool)

            print(f"\n---> [{r_name}] DR Object Evaluation ({len(episodes)} episodes, mode='{sample_mode}') <---\n")

            for ep_num, (obj_id, ep_idx) in enumerate(episodes, 1):
                if not viewer.is_running():
                    print("[Viewer] Window closed by user.")
                    break

                obj_name = OBJECT_NAMES[obj_id]
                seed = 90000 + (r_id * 1000) + (obj_id * 100) + ep_idx
                np.random.seed(seed)
                obs, _ = env.reset(object_id=obj_id)

                if obj_id == 1:
                    axis = np.array([np.random.uniform(-0.25, 0.25), np.random.uniform(-0.25, 0.25), np.random.choice([-1.0, 1.0])])
                    axis /= np.linalg.norm(axis)
                else:
                    axis = np.random.randn(3)
                    axis /= np.linalg.norm(axis) + 1e-8

                target_angle = float(np.random.uniform(min_rad, max_rad))
                rand_rot = transform.Rotation.from_rotvec(axis * target_angle)
                q_xyzw = rand_rot.as_quat()
                env.target_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

                obj_idx = env.model.jnt_qposadr[env.model.joint(f"obj_{obj_id}_joint").id]
                cur_obj_quat = env.data.qpos[obj_idx + 3 : obj_idx + 7]
                env.prev_d_theta = env.compute_angular_distance(cur_obj_quat, env.target_quat)
                env.initial_d_theta = float(env.prev_d_theta)

                init_angle_deg = float(np.degrees(env.initial_d_theta))
                tol_deg = float(epsilon_ratio * init_angle_deg)
                theta_rad = float(env.initial_d_theta)
                quad_steps = int(250 + 200 * theta_rad + 150.0 * (theta_rad**2))
                env.current_max_steps = min(800, quad_steps)
                env.set_epsilon_ratio(epsilon_ratio)
                env.set_epsilon(epsilon_ratio * theta_rad)

                # Refresh obs with new target
                obs = env._get_observation()

                half_height = 0.045 if obj_id == 2 else 0.07

                step = 0
                terminated = False
                truncated = False
                step_info = {}
                peak_torque = 0.0
                last_goals_count = 0
                smoothed_act = None

                active_lag_ms = env.active_action_delay * 20
                print(
                    f"[{r_name}] Ep {ep_num:02d}/{len(episodes)} | Object: {obj_name:<8} | "
                    f"Target: {init_angle_deg:.1f}° | Tol: {tol_deg:.1f}° | DR Delay: {active_lag_ms}ms"
                )

                while not (terminated or truncated) and viewer.is_running():
                    step += 1
                    with torch.no_grad():
                        raw_act = actor(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).numpy()
                    if smoothed_act is None:
                        smoothed_act = raw_act.copy()
                    else:
                        smoothed_act = smooth_alpha * raw_act + (1.0 - smooth_alpha) * smoothed_act
                    obs, reward, terminated, truncated, step_info = env.step(smoothed_act)

                    cur_torque = float(np.max(np.abs(env.data.actuator_force)))
                    if cur_torque > peak_torque:
                        peak_torque = cur_torque

                    cur_err_deg = float(np.degrees(step_info.get("d_theta", env.prev_d_theta)))
                    tol_deg = float(epsilon_ratio * init_angle_deg)

                    # Green orientation arrow anchored on object top surface
                    try:
                        obj_pos = env.data.qpos[obj_idx : obj_idx + 3].copy()
                        cur_q = env.data.qpos[obj_idx + 3 : obj_idx + 7]
                        cur_rot = transform.Rotation.from_quat([cur_q[1], cur_q[2], cur_q[3], cur_q[0]])
                        base_point = obj_pos + cur_rot.apply(np.array([0.0, 0.0, half_height + 0.005]))

                        tgt_q = env.target_quat
                        tgt_rot = transform.Rotation.from_quat([tgt_q[1], tgt_q[2], tgt_q[3], tgt_q[0]])
                        tgt_dir = tgt_rot.apply(np.array([0.0, 0.0, 1.0]))

                        arrow_len = 0.15
                        arrow_rad = 0.0035
                        tip_pos = base_point + arrow_len * tgt_dir

                        with viewer.lock():
                            viewer.user_scn.ngeom = 0
                            _add_user_sphere(viewer.user_scn, base_point, 0.0045, [1.0, 0.85, 0.0, 1.0])
                            _add_user_arrow(viewer.user_scn, base_point, tip_pos, arrow_rad, [0.05, 1.0, 0.05, 1.0])

                        viewer.set_texts([
                            (
                                mujoco.mjtFontScale.mjFONTSCALE_150,
                                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                f"[SIM-TO-REAL DR] {obj_name} Ep {ep_num}/{len(episodes)} | Delay: {active_lag_ms}ms | Noise: {obs_noise_scale:.2f}",
                                f"Err: {cur_err_deg:.1f}° (Tol: {tol_deg:.1f}°) | Goals: {step_info.get('successes_in_episode', 0)} | Step: {step}",
                            )
                        ])
                    except Exception:
                        pass

                    viewer.sync()
                    time.sleep(0.002)

                    current_goals = step_info.get("successes_in_episode", 0)
                    if current_goals > last_goals_count:
                        last_goals_count = current_goals
                        if current_goals >= max_goals:
                            terminated = True
                            break

                final_err_deg = float(np.degrees(step_info.get("d_theta", env.prev_d_theta)))
                goals_achieved = step_info.get("successes_in_episode", 0)
                is_success = bool(goals_achieved > 0)
                raw_dropped = bool(step_info.get("raw_dropped", False))

                if is_success and hold_time > 0:
                    hold_start = time.time()
                    last_ctrl = env.data.ctrl[:].copy()
                    while (time.time() - hold_start < hold_time) and viewer.is_running():
                        env.data.ctrl[:] = last_ctrl
                        env.data.qvel[:12] *= 0.85
                        mujoco.mj_step(env.model, env.data)
                        elapsed = time.time() - hold_start
                        viewer.set_texts([
                            (
                                mujoco.mjtFontScale.mjFONTSCALE_150,
                                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                f"*** TARGET REACHED UNDER DOMAIN RANDOMIZATION ***",
                                f"Holding Firm ({elapsed:.1f}s / {hold_time:.1f}s) | Error: {final_err_deg:.1f}° <= {epsilon_ratio*init_angle_deg:.1f}°",
                            )
                        ])
                        viewer.sync()
                        time.sleep(0.02)

                dropped_before_success = bool(raw_dropped and (goals_achieved == 0))
                pre_goal_retained = not dropped_before_success

                if goals_achieved > 0:
                    closure = 1.0
                elif dropped_before_success:
                    closure = 0.0
                else:
                    closure = max(0.0, 1.0 - (final_err_deg / (init_angle_deg + 1e-6)))

                rec = {
                    "range_id": r_id,
                    "object_id": obj_id,
                    "init_angle_deg": init_angle_deg,
                    "final_err_deg": final_err_deg,
                    "is_success": is_success,
                    "closure": closure,
                    "raw_dropped": raw_dropped,
                    "pre_goal_retained": pre_goal_retained,
                    "dropped_before_success": dropped_before_success,
                    "goals_achieved": goals_achieved,
                    "steps": step,
                    "peak_torque": peak_torque,
                    "action_delay_steps": getattr(env, "active_action_delay", 0),
                }
                all_results.append(rec)

                succ_tag = "SUCCESS" if is_success else "FAIL"
                drop_tag = "DROPPED" if raw_dropped else "HELD"
                print(
                    f"  Ep {ep_num:03d} | Init: {init_angle_deg:4.1f}° | Err: {final_err_deg:4.1f}° | "
                    f"Clos: {closure*100:5.1f}% | {succ_tag:<7} | {drop_tag:<7} | Steps: {step:3d} | Delay: {active_lag_ms}ms"
                )
            if not viewer.is_running():
                break

    dr_info = {
        "action_delay_max": action_delay_max,
        "obs_noise_scale": obs_noise_scale,
        "dr_ratio": dr_ratio,
    }
    generate_metric_report(all_results, epsilon_ratio, checkpoint_path, dr_info, report_file)


def run_parallel_evaluation(
    checkpoint_path: str,
    scene_xml_path: str,
    episodes_per_range: int = 100,
    num_workers: int = 16,
    epsilon_ratio: float = 0.10,
    smooth_alpha: float = 0.50,
    sample_mode: str = "shuffle",
    action_delay_max: int = 1,
    vision_delay_max: int = 2,
    obs_noise_scale: float = 0.80,
    dr_ratio: float = 0.30,
    report_file: str = "EVALUATION_DR_REPORT.md",
):
    """High-speed headless multi-worker evaluation under active Domain Randomization."""
    print(f"\n[Domain Randomization Evaluator] Loading Checkpoint: {checkpoint_path}")
    print(f"  Sim-to-Real DR Parameters:")
    print(f"    - Actuation Latency Buffer : 0 to {action_delay_max} steps (0 - {action_delay_max*20} ms lag)")
    print(f"    - ArUco Vision Latency     : 1 to {vision_delay_max} steps (20 - {vision_delay_max*20} ms lag)")
    print(f"    - Action Smooth Alpha      : {smooth_alpha:.2f}")
    print(f"    - Observation Sensor Noise : Scale {obs_noise_scale:.2f} (encoder, velocity, vision jitter)")
    print(f"    - Silicone Friction Band   : μ in [2.1, 3.9] (±30% dynamic sliding friction)")
    print(f"    - Object Property Variation: ±{int(dr_ratio*100)}% (mass, friction, geometry)")
    print(f"    - Privileged Access        : DISABLED (nominal physical observations only)")
    print(f"  Protocol: {episodes_per_range} episodes per range × 2 ranges × 3 objects = {episodes_per_range * 6} total episodes")
    print(f"  Target Precision Envelope: {epsilon_ratio*100:.1f}% relative error tolerance")
    print(f"  Workers: {num_workers} parallel CPU processes | Smooth Alpha: {smooth_alpha:.2f} | Sample Mode: {sample_mode}\n")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("actor_state_dict", payload.get("model_state_dict"))
    obs_mean = payload["obs_mean"]
    obs_std = payload["obs_std"]

    tasks = []
    for r_cfg in RANGE_DEFINITIONS:
        r_id = r_cfg["id"]
        min_rad = r_cfg["min_rad"]
        max_rad = r_cfg["max_rad"]
        if sample_mode == "sequential":
            for obj_id in [1, 2, 3]:
                for ep in range(episodes_per_range):
                    seed = 60000 + (r_id * 1000) + (obj_id * 100) + ep
                    tasks.append((seed, r_id, obj_id, min_rad, max_rad))
        elif sample_mode == "random":
            total_eps = episodes_per_range * 3
            for ep in range(total_eps):
                obj_id = int(np.random.choice([1, 2, 3]))
                seed = 60000 + (r_id * 1000) + ep
                tasks.append((seed, r_id, obj_id, min_rad, max_rad))
        else:
            range_tasks = []
            for obj_id in [1, 2, 3]:
                for ep in range(episodes_per_range):
                    seed = 60000 + (r_id * 1000) + (obj_id * 100) + ep
                    range_tasks.append((seed, r_id, obj_id, min_rad, max_rad))
            tasks.extend(_shuffle_avoid_adjacent_repeats(range_tasks))

    t_start = time.time()
    init_args = (
        state_dict,
        obs_mean,
        obs_std,
        scene_xml_path,
        epsilon_ratio,
        smooth_alpha,
        action_delay_max,
        vision_delay_max,
        obs_noise_scale,
        dr_ratio,
    )
    ctx = mp.get_context("spawn")

    print(f"Starting parallel DR execution of {len(tasks)} benchmark episodes...")
    with ctx.Pool(num_workers, initializer=_worker_init, initargs=init_args) as pool:
        results = pool.map(_eval_single_task, tasks)

    elapsed = time.time() - t_start
    print(f"Completed all {len(results)} DR episodes in {elapsed:.1f}s ({len(results)/elapsed:.1f} eps/sec).\n")

    dr_info = {
        "action_delay_max": action_delay_max,
        "vision_delay_max": vision_delay_max,
        "smooth_alpha": smooth_alpha,
        "obs_noise_scale": obs_noise_scale,
        "dr_ratio": dr_ratio,
    }
    return generate_metric_report(results, epsilon_ratio, checkpoint_path, dr_info, report_file)


def compute_group_metrics(subset):
    n = len(subset)
    if n == 0:
        return {}
    succ_rate = sum(1 for r in subset if r["is_success"]) / n * 100
    clos_mean = np.mean([r["closure"] for r in subset]) * 100
    pre_ret_rate = sum(1 for r in subset if r["pre_goal_retained"]) / n * 100
    pre_drop_rate = sum(1 for r in subset if r["dropped_before_success"]) / n * 100
    raw_drop_rate = sum(1 for r in subset if r["raw_dropped"]) / n * 100
    final_err_mean = np.mean([r["final_err_deg"] for r in subset])
    init_ang_mean = np.mean([r["init_angle_deg"] for r in subset])
    mean_steps = np.mean([r["steps"] for r in subset])
    peak_torque = max([r["peak_torque"] for r in subset]) if subset else 0.0
    emi = (pre_ret_rate / 100.0) * (clos_mean / 100.0)
    score = 0.5 * (succ_rate / 100.0) + 0.3 * (clos_mean / 100.0) + 0.2 * (pre_ret_rate / 100.0)

    return {
        "n": n,
        "succ_rate": succ_rate,
        "clos_mean": clos_mean,
        "pre_ret_rate": pre_ret_rate,
        "pre_drop_rate": pre_drop_rate,
        "raw_drop_rate": raw_drop_rate,
        "final_err_mean": final_err_mean,
        "init_ang_mean": init_ang_mean,
        "mean_steps": mean_steps,
        "peak_torque": peak_torque,
        "emi": emi,
        "score": score,
    }


def generate_metric_report(results, epsilon_ratio: float, checkpoint_path: str, dr_info: dict, report_file: str):
    """Prints comprehensive markdown and ASCII metric tables."""
    if not results:
        print("No evaluation records collected.")
        return ""

    action_delay_max = dr_info.get("action_delay_max", 1)
    vision_delay_max = dr_info.get("vision_delay_max", 2)
    smooth_alpha = dr_info.get("smooth_alpha", 0.50)
    obs_noise_scale = dr_info.get("obs_noise_scale", 0.80)
    dr_ratio = dr_info.get("dr_ratio", 0.30)

    lines = []
    lines.append("=" * 86)
    lines.append(f"IN-HAND MANIPULATION SIM-TO-REAL DOMAIN RANDOMIZATION REPORT (@{epsilon_ratio*100:.1f}% Tol)")
    lines.append(f"Checkpoint: {checkpoint_path}")
    lines.append(f"Total Episodes Evaluated: {len(results)}")
    lines.append(f"DR Conditions:")
    lines.append(f"  • Actuation Delay: 0 to {action_delay_max} steps (0 to {action_delay_max*20} ms lag)")
    lines.append(f"  • ArUco Vision Latency: 1 to {vision_delay_max} steps (20 to {vision_delay_max*20} ms camera+pose lag)")
    lines.append(f"  • Action Smoothing Filter Alpha: {smooth_alpha:.2f}")
    lines.append(f"  • Observation Sensor Noise Scale: {obs_noise_scale:.2f}")
    lines.append(f"  • Fingertip/Palm Silicone Friction: μ in [2.1, 3.9]")
    lines.append(f"  • Object Geometry / Mass / Friction: ±{int(dr_ratio*100)}%")
    lines.append(f"  • Ground-Truth Physical Privileges: Disabled (Nominal Observation Only)")
    lines.append("=" * 86)

    # 1. RANGE 1 REPORT: 10° to 40°
    r1_results = [r for r in results if r["range_id"] == 1]
    lines.append("\n" + "#" * 86)
    lines.append("SECTION 1: RANGE 1 (10.0° to 40.0° Small-to-Medium Rotations under DR)")
    lines.append("#" * 86)
    header = f"{'Object':<12} | {'Episodes':<8} | {'Success':<9} | {'Closure':<9} | {'Pre-Ret':<9} | {'Pre-Drop':<9} | {'Raw-Drop':<9} | {'Final Err':<10} | {'Steps':<6}"
    lines.append(header)
    lines.append("-" * len(header))

    for obj_id in [1, 2, 3]:
        sub = [r for r in r1_results if r["object_id"] == obj_id]
        m = compute_group_metrics(sub)
        if m:
            lines.append(
                f"{OBJECT_NAMES[obj_id]:<12} | {m['n']:<8d} | {m['succ_rate']:6.1f}%  | {m['clos_mean']:6.1f}%  | "
                f"{m['pre_ret_rate']:6.1f}%  | {m['pre_drop_rate']:6.1f}%  | {m['raw_drop_rate']:6.1f}%  | {m['final_err_mean']:5.1f}°     | {m['mean_steps']:5.1f}"
            )

    r1_total = compute_group_metrics(r1_results)
    lines.append("-" * len(header))
    lines.append(
        f"{'RANGE 1 TOTAL':<12} | {r1_total['n']:<8d} | {r1_total['succ_rate']:6.1f}%  | {r1_total['clos_mean']:6.1f}%  | "
        f"{r1_total['pre_ret_rate']:6.1f}%  | {r1_total['pre_drop_rate']:6.1f}%  | {r1_total['raw_drop_rate']:6.1f}%  | {r1_total['final_err_mean']:5.1f}°     | {r1_total['mean_steps']:5.1f}"
    )
    lines.append(f"  --> Range 1 Effective Manipulation Index (EMI): {r1_total['emi']:.4f} | Benchmark Score: {r1_total['score']:.4f}")

    # 2. RANGE 2 REPORT: 40° to 90°
    r2_results = [r for r in results if r["range_id"] == 2]
    lines.append("\n" + "#" * 86)
    lines.append("SECTION 2: RANGE 2 (40.0° to 90.0° Large-Angle In-Hand Gaiting under DR)")
    lines.append("#" * 86)
    lines.append(header)
    lines.append("-" * len(header))

    for obj_id in [1, 2, 3]:
        sub = [r for r in r2_results if r["object_id"] == obj_id]
        m = compute_group_metrics(sub)
        if m:
            lines.append(
                f"{OBJECT_NAMES[obj_id]:<12} | {m['n']:<8d} | {m['succ_rate']:6.1f}%  | {m['clos_mean']:6.1f}%  | "
                f"{m['pre_ret_rate']:6.1f}%  | {m['pre_drop_rate']:6.1f}%  | {m['raw_drop_rate']:6.1f}%  | {m['final_err_mean']:5.1f}°     | {m['mean_steps']:5.1f}"
            )

    r2_total = compute_group_metrics(r2_results)
    lines.append("-" * len(header))
    lines.append(
        f"{'RANGE 2 TOTAL':<12} | {r2_total['n']:<8d} | {r2_total['succ_rate']:6.1f}%  | {r2_total['clos_mean']:6.1f}%  | "
        f"{r2_total['pre_ret_rate']:6.1f}%  | {r2_total['pre_drop_rate']:6.1f}%  | {r2_total['raw_drop_rate']:6.1f}%  | {r2_total['final_err_mean']:5.1f}°     | {r2_total['mean_steps']:5.1f}"
    )
    lines.append(f"  --> Range 2 Effective Manipulation Index (EMI): {r2_total['emi']:.4f} | Benchmark Score: {r2_total['score']:.4f}")

    # 3. GRAND COMPOSITE SUMMARY REPORT
    grand_total = compute_group_metrics(results)
    lines.append("\n" + "=" * 86)
    lines.append("SECTION 3: GRAND COMPOSITE SIM-TO-REAL DR SUMMARY (FULL 10° TO 90° ENVELOPE)")
    lines.append("=" * 86)
    lines.append(f"{'Evaluation Metric':<35} | {'Range 1 (10°-40°)':<20} | {'Range 2 (40°-90°)':<20} | {'Grand Total':<15}")
    lines.append("-" * 96)
    lines.append(f"{'Total Evaluated Episodes':<35} | {r1_total['n']:<20d} | {r2_total['n']:<20d} | {grand_total['n']:<15d}")
    lines.append(f"{f'Success Rate (@{epsilon_ratio*100:.1f}% Error)':<35} | {r1_total['succ_rate']:<19.1f}% | {r2_total['succ_rate']:<19.1f}% | {grand_total['succ_rate']:<14.1f}%")
    lines.append(f"{'Mean Rotational Closure':<35} | {r1_total['clos_mean']:<19.1f}% | {r2_total['clos_mean']:<19.1f}% | {grand_total['clos_mean']:<14.1f}%")
    lines.append(f"{'Pre-Goal Retention Rate':<35} | {r1_total['pre_ret_rate']:<19.1f}% | {r2_total['pre_ret_rate']:<19.1f}% | {grand_total['pre_ret_rate']:<14.1f}%")
    lines.append(f"{'Pre-Goal Drop Rate':<35} | {r1_total['pre_drop_rate']:<19.1f}% | {r2_total['pre_drop_rate']:<19.1f}% | {grand_total['pre_drop_rate']:<14.1f}%")
    lines.append(f"{'Raw Drop Rate (All Steps)':<35} | {r1_total['raw_drop_rate']:<19.1f}% | {r2_total['raw_drop_rate']:<19.1f}% | {grand_total['raw_drop_rate']:<14.1f}%")
    lines.append(f"{'Mean Final Geodesic Error':<35} | {r1_total['final_err_mean']:<19.1f}° | {r2_total['final_err_mean']:<19.1f}° | {grand_total['final_err_mean']:<14.1f}°")
    lines.append(f"{'Mean Steps per Episode':<35} | {r1_total['mean_steps']:<19.1f}  | {r2_total['mean_steps']:<19.1f}  | {grand_total['mean_steps']:<14.1f} ")
    lines.append(f"{'Peak Hardware Torque':<35} | {r1_total['peak_torque']:<17.2f}N·m | {r2_total['peak_torque']:<17.2f}N·m | {grand_total['peak_torque']:<12.2f}N·m")
    lines.append(f"{'Effective Manipulation Index (EMI)':<35} | {r1_total['emi']:<20.4f} | {r2_total['emi']:<20.4f} | {grand_total['emi']:<15.4f}")
    lines.append(f"{'Composite Benchmark Score':<35} | {r1_total['score']:<20.4f} | {r2_total['score']:<20.4f} | {grand_total['score']:<15.4f}")
    lines.append("=" * 96 + "\n")

    report_str = "\n".join(lines)
    print(report_str)

    # Save to report file in root or custom destination
    target_path = report_file if os.path.isabs(report_file) else os.path.join(PROJECT_ROOT, report_file)
    with open(target_path, "w") as f:
        f.write(report_str + "\n")

    print(f"[Domain Randomization Report Saved]: {target_path}")
    return report_str


def main():
    parser = argparse.ArgumentParser(
        description="Domain Randomization (Sim-to-Real) Benchmark Evaluator (Delto Gripper-3F)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(PROJECT_ROOT, "best_model", "best_model.pth"),
        help="Path to actor checkpoint (.pth)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open interactive 3D MuJoCo passive viewer with live DR telemetry HUD",
    )
    parser.add_argument(
        "--episodes_per_range",
        type=int,
        default=100,
        help="Number of episodes per range per object type (default: 100 -> 600 total)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of CPU worker processes in parallel mode (default: 16)",
    )
    parser.add_argument(
        "--epsilon_ratio",
        type=float,
        default=0.10,
        help="Relative error tolerance for success (default: 0.10, i.e. 10%% error)",
    )
    parser.add_argument(
        "--action_delay_max",
        type=int,
        default=1,
        help="Maximum action delay steps (default: 1 -> 0 to 20ms delay queue matching real CAN bus)",
    )
    parser.add_argument(
        "--vision_delay_max",
        type=int,
        default=2,
        help="Maximum ArUco vision delay steps (default: 2 -> 20 to 40ms camera capture + pose estimation latency)",
    )
    parser.add_argument(
        "--obs_noise_scale",
        type=float,
        default=0.80,
        help="Observation sensor noise multiplier (default: 0.80)",
    )
    parser.add_argument(
        "--dr_ratio",
        type=float,
        default=0.30,
        help="Domain randomization ratio on mass, friction, size (default: 0.30 for ±30%%)",
    )
    parser.add_argument(
        "--scene_xml",
        type=str,
        default=os.path.join(PROJECT_ROOT, "in_hand_manipulation", "simulation", "scene_cube.xml"),
        help="Path to scene XML model",
    )
    parser.add_argument(
        "--hold_time",
        type=float,
        default=1.2,
        help="Hold duration in seconds once goal is achieved during presentation (default: 1.2s)",
    )
    parser.add_argument(
        "--max_goals",
        type=int,
        default=1,
        help="Target goals to complete per episode in visual demo mode before next episode (default: 1)",
    )
    parser.add_argument(
        "--smooth_alpha",
        type=float,
        default=0.50,
        help="Action filter alpha (default: 0.50 exponential smoothing filter)",
    )
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="shuffle",
        choices=["shuffle", "random", "sequential"],
        help="Object sampling mode across episodes: 'shuffle' (randomized stratified permutation, default)",
    )
    default_report = os.path.join(PROJECT_ROOT, "EVAL", "EVALUATION_DR_REPORT.md") if os.path.isdir(os.path.join(PROJECT_ROOT, "EVAL")) else "EVALUATION_DR_REPORT.md"
    parser.add_argument(
        "--report_file",
        type=str,
        default=default_report,
        help="Output report markdown file path (default: EVAL/EVALUATION_DR_REPORT.md)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible episode ordering and sampling.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at: {args.checkpoint}")
        sys.exit(1)

    if args.render:
        run_rendered_evaluation(
            checkpoint_path=args.checkpoint,
            scene_xml_path=args.scene_xml,
            episodes_per_range=args.episodes_per_range,
            epsilon_ratio=args.epsilon_ratio,
            action_delay_max=args.action_delay_max,
            vision_delay_max=args.vision_delay_max,
            obs_noise_scale=args.obs_noise_scale,
            dr_ratio=args.dr_ratio,
            hold_time=args.hold_time,
            max_goals=args.max_goals,
            smooth_alpha=args.smooth_alpha,
            sample_mode=args.sample_mode,
            report_file=args.report_file,
        )
    else:
        run_parallel_evaluation(
            checkpoint_path=args.checkpoint,
            scene_xml_path=args.scene_xml,
            episodes_per_range=args.episodes_per_range,
            num_workers=args.num_workers,
            epsilon_ratio=args.epsilon_ratio,
            smooth_alpha=args.smooth_alpha,
            sample_mode=args.sample_mode,
            action_delay_max=args.action_delay_max,
            vision_delay_max=args.vision_delay_max,
            obs_noise_scale=args.obs_noise_scale,
            dr_ratio=args.dr_ratio,
            report_file=args.report_file,
        )


if __name__ == "__main__":
    main()


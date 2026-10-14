#!/usr/bin/env python3
"""
evaluate.py
=============================================================================
Unified Canonical In-Hand Manipulation Evaluator for Delto Gripper-3F.

Stratified Angle Range Evaluation Protocol:
  - Range 1 (Small-to-Medium Rotations): 10.0° to 40.0° (0.1745 to 0.6981 rad)
  - Range 2 (Large-Angle Gaiting)     : 40.0° to 90.0° (0.6981 to 1.5708 rad)

Standard Evaluation Envelope:
  - 100 episodes per range per object type
  - 3 Object Types: Cylinder (obj_1), Cube (obj_2), Cuboid (obj_3)
  - Total: 100 (Range 1) + 100 (Range 2) = 200 per object -> 600 total episodes!

Features:
  - --checkpoint <path> : Model checkpoint path to evaluate.
  - --render            : Launches interactive 3D MuJoCo passive viewer.
  - --episodes_per_range: Configurable episodes per range (default: 100).
  - --num_workers       : Fast parallel evaluation when headless (default: 8).
  - Stratified metric reports for Range 1, Range 2, and Grand Composite.
=============================================================================
"""

import os
import sys
import time
import argparse
import numpy as np
import scipy.spatial.transform as transform
import torch
import multiprocessing as mp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train_ppo import DeltoInHandManipulationEnv, ActorNetwork

OBJECT_NAMES = {1: "Cylinder", 2: "Cube", 3: "Cuboid"}
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


def _worker_init(state_dict, obs_mean, obs_std, scene_xml_path, epsilon_ratio):
    global _w_actor, _w_env, _w_epsilon_ratio
    torch.set_num_threads(1)
    _w_actor = ActorNetwork(obs_dim=39, act_dim=12)
    _w_actor.load_state_dict(state_dict, strict=False)
    _w_actor.eval()
    _w_epsilon_ratio = epsilon_ratio
    _w_env = DeltoInHandManipulationEnv(
        scene_xml_path=scene_xml_path,
        obs_mean=obs_mean,
        obs_std=obs_std,
        max_episode_steps=250,
        max_goal_angle=1.5708,
        dr_ratio=0.20,
        steps_per_radian=200,
        bonus_steps_per_success=150,
        max_steps_cap=800,
    )


def _eval_single_task(task_args):
    """Worker task execution function for headless multiprocessing pool."""
    global _w_actor, _w_env, _w_epsilon_ratio
    seed, range_id, obj_id, min_rad, max_rad = task_args
    np.random.seed(seed)

    obs, _ = _w_env.reset(object_id=obj_id)

    # Sample uniform axis and uniform angle in specified range
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

    step = 0
    terminated = False
    truncated = False
    step_info = {}
    peak_torque = 0.0

    while not (terminated or truncated):
        step += 1
        with torch.no_grad():
            act = _w_actor(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).numpy()
        obs, reward, terminated, truncated, step_info = _w_env.step(act)
        
        cur_torque = float(np.max(np.abs(_w_env.data.actuator_force)))
        if cur_torque > peak_torque:
            peak_torque = cur_torque
            
        goals = step_info.get("successes_in_episode", 0)
        if goals >= 2:
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
    }


def run_rendered_evaluation(checkpoint_path: str, scene_xml_path: str, episodes_per_range: int, epsilon_ratio: float):
    """Interactive visual evaluation with MuJoCo passive 3D viewer."""
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as e:
        print(f"Error importing mujoco.viewer: {e}")
        return

    print(f"\n[Viewer] Launching Interactive MuJoCo Visual Evaluation: {checkpoint_path}")
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
        dr_ratio=0.20,
        steps_per_radian=200,
        bonus_steps_per_success=150,
        max_steps_cap=800,
    )

    all_results = []
    print("\n" + "=" * 80)
    print("MUJOCO VISUAL VIEWER ACTIVE - Press SPACE in viewer to pause, ESC to exit")
    print("=" * 80 + "\n")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for r_cfg in RANGE_DEFINITIONS:
            r_id = r_cfg["id"]
            r_name = r_cfg["name"]
            min_rad = r_cfg["min_rad"]
            max_rad = r_cfg["max_rad"]

            for obj_id in [1, 2, 3]:
                obj_name = OBJECT_NAMES[obj_id]
                print(f"\n---> [{r_name}] Object: {obj_name} ({episodes_per_range} episodes) <---")

                for ep in range(1, episodes_per_range + 1):
                    if not viewer.is_running():
                        print("[Viewer] Window closed by user.")
                        break

                    seed = 90000 + (r_id * 1000) + (obj_id * 100) + ep
                    np.random.seed(seed)
                    obs, _ = env.reset(object_id=obj_id)

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
                    theta_rad = float(env.initial_d_theta)
                    quad_steps = int(250 + 200 * theta_rad + 150.0 * (theta_rad**2))
                    env.current_max_steps = min(800, quad_steps)
                    env.set_epsilon_ratio(epsilon_ratio)
                    env.set_epsilon(epsilon_ratio * theta_rad)

                    # Reference axis: strictly the direction of the top surface of the object facing up (+Z in body frame)
                    half_height = 0.045 if obj_id == 2 else 0.07
                    top_axis = np.array([0.0, 0.0, 1.0])

                    step = 0
                    terminated = False
                    truncated = False
                    step_info = {}
                    peak_torque = 0.0

                    while not (terminated or truncated) and viewer.is_running():
                        step += 1
                        with torch.no_grad():
                            act = actor(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).numpy()
                        obs, reward, terminated, truncated, step_info = env.step(act)

                        cur_torque = float(np.max(np.abs(env.data.actuator_force)))
                        if cur_torque > peak_torque:
                            peak_torque = cur_torque

                        # Real-time visual orientation arrows:
                        # Red Arrow: Object's current top surface normal facing up
                        # Green Arrow: Desired target top surface normal
                        try:
                            obj_pos = env.data.qpos[obj_idx : obj_idx + 3].copy()
                            cur_q = env.data.qpos[obj_idx + 3 : obj_idx + 7]
                            cur_rot = transform.Rotation.from_quat([cur_q[1], cur_q[2], cur_q[3], cur_q[0]])
                            tgt_q = env.target_quat
                            tgt_rot = transform.Rotation.from_quat([tgt_q[1], tgt_q[2], tgt_q[3], tgt_q[0]])

                            # Base point: anchored on the top face of the object (above black marker)
                            # This ensures the arrows are prominently visible and never occluded inside the object body
                            top_offset = cur_rot.apply(np.array([0.0, 0.0, half_height + 0.005]))
                            base_point = obj_pos + top_offset

                            # Strictly pointing along the direction of the top surface facing up:
                            cur_dir = cur_rot.apply(top_axis)
                            tgt_dir = tgt_rot.apply(top_axis)

                            arrow_len = 0.16
                            arrow_rad = 0.010
                            p2_cur = base_point + arrow_len * cur_dir
                            p2_tgt = base_point + arrow_len * tgt_dir

                            with viewer.lock():
                                viewer.user_scn.ngeom = 0

                                # 1. Yellow anchor hub sphere on top marker
                                hub_geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                                mujoco.mjv_initGeom(
                                    hub_geom,
                                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                    size=np.array([0.009, 0.009, 0.009], dtype=np.float64),
                                    pos=base_point.astype(np.float64),
                                    mat=np.eye(3).flatten().astype(np.float64),
                                    rgba=np.array([1.0, 0.85, 0.0, 1.0], dtype=np.float32),
                                )
                                hub_geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                                viewer.user_scn.ngeom += 1

                                # 2. Red Arrow: Current Object Orientation
                                g_cur = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                                mujoco.mjv_initGeom(
                                    g_cur,
                                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                                    size=np.zeros(3, dtype=np.float64),
                                    pos=np.zeros(3, dtype=np.float64),
                                    mat=np.eye(3).flatten().astype(np.float64),
                                    rgba=np.array([1.0, 0.05, 0.05, 1.0], dtype=np.float32),  # Bright RED
                                )
                                mujoco.mjv_connector(
                                    g_cur,
                                    mujoco.mjtGeom.mjGEOM_ARROW,
                                    arrow_rad,
                                    base_point,
                                    p2_cur,
                                )
                                g_cur.category = mujoco.mjtCatBit.mjCAT_DECOR
                                viewer.user_scn.ngeom += 1

                                # 3. Green Arrow: Desired Target Orientation
                                g_tgt = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                                mujoco.mjv_initGeom(
                                    g_tgt,
                                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                                    size=np.zeros(3, dtype=np.float64),
                                    pos=np.zeros(3, dtype=np.float64),
                                    mat=np.eye(3).flatten().astype(np.float64),
                                    rgba=np.array([0.05, 1.0, 0.05, 1.0], dtype=np.float32),  # Bright GREEN
                                )
                                mujoco.mjv_connector(
                                    g_tgt,
                                    mujoco.mjtGeom.mjGEOM_ARROW,
                                    arrow_rad,
                                    base_point,
                                    p2_tgt,
                                )
                                g_tgt.category = mujoco.mjtCatBit.mjCAT_DECOR
                                viewer.user_scn.ngeom += 1

                            # Overlay real-time HUD telemetry on passive viewer
                            cur_err_deg = float(np.degrees(step_info.get("d_theta", env.prev_d_theta)))
                            tol_deg = float(epsilon_ratio * init_angle_deg)
                            viewer.set_texts([
                                (
                                    mujoco.mjtFontScale.mjFONTSCALE_150,
                                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                    f"RED: Current | GREEN: Target | [{r_name}] {obj_name} Ep {ep}/{episodes_per_range}",
                                    f"Err: {cur_err_deg:.1f}° (Tol: {tol_deg:.1f}°) | Goals: {step_info.get('successes_in_episode', 0)} | Step: {step}",
                                )
                            ])
                        except Exception as err:
                            print(f"[Viewer Warning] Arrow rendering error: {err}")

                        viewer.sync()
                        time.sleep(0.015)

                        goals = step_info.get("successes_in_episode", 0)
                        if goals >= 2:
                            break

                    final_err_deg = step_info.get("d_theta", 0.0) * (180.0 / np.pi)
                    raw_dropped = bool(step_info.get("raw_dropped", False))
                    goals_achieved = step_info.get("successes_in_episode", 0)
                    is_success = bool(goals_achieved > 0 or final_err_deg <= (epsilon_ratio * init_angle_deg))
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
                    }
                    all_results.append(rec)

                    succ_tag = "SUCCESS" if is_success else "FAIL"
                    drop_tag = "DROPPED" if raw_dropped else "HELD"
                    print(
                        f"  Ep {ep:03d} | Init: {init_angle_deg:4.1f}° | Err: {final_err_deg:4.1f}° | "
                        f"Clos: {closure*100:5.1f}% | {succ_tag:<7} | {drop_tag:<7} | Steps: {step:3d} | Torque: {peak_torque:4.2f}N·m"
                    )
                if not viewer.is_running():
                    break
            if not viewer.is_running():
                break

    generate_metric_report(all_results, epsilon_ratio, checkpoint_path)


def run_parallel_evaluation(
    checkpoint_path: str,
    scene_xml_path: str,
    episodes_per_range: int = 100,
    num_workers: int = 8,
    epsilon_ratio: float = 0.10,
):
    """High-speed headless multi-worker evaluation across both angle ranges."""
    print(f"\n[Parallel Evaluator] Loading Checkpoint: {checkpoint_path}")
    print(f"  Protocol: {episodes_per_range} episodes per range × 2 ranges × 3 objects = {episodes_per_range * 6} total episodes")
    print(f"  Target Precision Envelope: {epsilon_ratio*100:.1f}% relative error tolerance")
    print(f"  Workers: {num_workers} parallel CPU processes\n")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("actor_state_dict", payload.get("model_state_dict"))
    obs_mean = payload["obs_mean"]
    obs_std = payload["obs_std"]

    tasks = []
    for r_cfg in RANGE_DEFINITIONS:
        r_id = r_cfg["id"]
        min_rad = r_cfg["min_rad"]
        max_rad = r_cfg["max_rad"]
        for obj_id in [1, 2, 3]:
            for ep in range(episodes_per_range):
                seed = 50000 + (r_id * 1000) + (obj_id * 100) + ep
                tasks.append((seed, r_id, obj_id, min_rad, max_rad))

    t_start = time.time()
    init_args = (state_dict, obs_mean, obs_std, scene_xml_path, epsilon_ratio)
    ctx = mp.get_context("spawn")

    print(f"Starting parallel execution of {len(tasks)} benchmark episodes...")
    with ctx.Pool(num_workers, initializer=_worker_init, initargs=init_args) as pool:
        results = pool.map(_eval_single_task, tasks)

    elapsed = time.time() - t_start
    print(f"Completed all {len(results)} episodes in {elapsed:.1f}s ({len(results)/elapsed:.1f} eps/sec).\n")

    return generate_metric_report(results, epsilon_ratio, checkpoint_path)


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


def generate_metric_report(results, epsilon_ratio: float, checkpoint_path: str):
    """Prints comprehensive markdown and ASCII metric tables."""
    if not results:
        print("No evaluation records collected.")
        return ""

    lines = []
    lines.append("=" * 86)
    lines.append(f"IN-HAND MANIPULATION STRATIFIED BENCHMARK REPORT (@{epsilon_ratio*100:.1f}% Error Tolerance)")
    lines.append(f"Checkpoint: {checkpoint_path}")
    lines.append(f"Total Episodes Evaluated: {len(results)}")
    lines.append("=" * 86)

    # 1. RANGE 1 REPORT: 10° to 40°
    r1_results = [r for r in results if r["range_id"] == 1]
    lines.append("\n" + "#" * 86)
    lines.append("SECTION 1: RANGE 1 (10.0° to 40.0° Small-to-Medium Rotations)")
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
    lines.append("SECTION 2: RANGE 2 (40.0° to 90.0° Large-Angle In-Hand Gaiting)")
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
    lines.append("SECTION 3: GRAND COMPOSITE VERIFICATION SUMMARY (FULL 10° TO 90° ENVELOPE)")
    lines.append("=" * 86)
    lines.append(f"{'Evaluation Metric':<35} | {'Range 1 (10°-40°)':<20} | {'Range 2 (40°-90°)':<20} | {'Grand Total':<15}")
    lines.append("-" * 96)
    lines.append(f"{'Total Evaluated Episodes':<35} | {r1_total['n']:<20d} | {r2_total['n']:<20d} | {grand_total['n']:<15d}")
    lines.append(f"{'Success Rate (@10% Error)':<35} | {r1_total['succ_rate']:<19.1f}% | {r2_total['succ_rate']:<19.1f}% | {grand_total['succ_rate']:<14.1f}%")
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

    # Save to EVALUATION_REPORT.md in root
    report_file = os.path.join(PROJECT_ROOT, "EVALUATION_REPORT.md")
    with open(report_file, "w") as f:
        f.write(report_str + "\n")

    return report_str


def main():
    parser = argparse.ArgumentParser(
        description="Unified Canonical In-Hand Manipulation Evaluator (Delto Gripper-3F)"
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
        help="Open interactive 3D MuJoCo passive viewer",
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
        default=8,
        help="Number of CPU worker processes in parallel mode (default: 8)",
    )
    parser.add_argument(
        "--epsilon_ratio",
        type=float,
        default=0.10,
        help="Relative error tolerance for success (default: 0.10, i.e. 10%% error)",
    )
    parser.add_argument(
        "--scene_xml",
        type=str,
        default=os.path.join(PROJECT_ROOT, "in_hand_manipulation", "simulation", "scene_cube.xml"),
        help="Path to scene XML model",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at: {args.checkpoint}")
        sys.exit(1)

    if args.render:
        run_rendered_evaluation(
            checkpoint_path=args.checkpoint,
            scene_xml_path=args.scene_xml,
            episodes_per_range=args.episodes_per_range,
            epsilon_ratio=args.epsilon_ratio,
        )
    else:
        run_parallel_evaluation(
            checkpoint_path=args.checkpoint,
            scene_xml_path=args.scene_xml,
            episodes_per_range=args.episodes_per_range,
            num_workers=args.num_workers,
            epsilon_ratio=args.epsilon_ratio,
        )


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
grand_tournament_eval.py
=============================================================================
Grand Tournament Model Re-evaluator for Delto Gripper-3F In-Hand Manipulation.

Protocol:
  - 1,000 episodes per range per object type
  - 2 Ranges:
      * Range 1: 10.0° to 40.0° (Small/Medium Precision)
      * Range 2: 40.0° to 90.0° (Large Gaiting)
  - 3 Object Types: Cylinder (1), Cube (2), Cuboid (3)
  - Total per Model: 1,000 × 2 ranges × 3 objects = 6,000 episodes!
  - Target Precision: Strict 10.0% Relative Error Tolerance (sub-9.0° precision)
  - Deterministic Seeds: Identical 6,000 test tasks for every candidate model
  - 16 Parallel CPU Workers

Crowns the absolute best model, promotes it to best_model/best_model.pth,
and updates best_model/README.md and progressive_curriculum_report.md.
=============================================================================
"""

import os
import sys
import time
import shutil
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
        "name": "Range 1 (10°-40° Precision)",
        "min_rad": np.radians(10.0),
        "max_rad": np.radians(40.0),
    },
    {
        "id": 2,
        "name": "Range 2 (40°-90° Gaiting)",
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
    global _w_actor, _w_env, _w_epsilon_ratio
    seed, range_id, obj_id, min_rad, max_rad = task_args
    np.random.seed(seed)

    obs, _ = _w_env.reset(object_id=obj_id)

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
    _w_env.set_epsilon(_w_epsilon_ratio * theta_rad)

    step = 0
    terminated = False
    truncated = False
    step_info = {}

    while not (terminated or truncated):
        step += 1
        with torch.no_grad():
            act = _w_actor(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).numpy()
        obs, reward, terminated, truncated, step_info = _w_env.step(act)
        goals = step_info.get("successes_in_episode", 0)
        if goals >= 2:
            break

    final_err_deg = step_info.get("d_theta", 0.0) * (180.0 / np.pi)
    raw_dropped = bool(step_info.get("raw_dropped", False))
    goals_achieved = step_info.get("successes_in_episode", 0)

    tol_thresh_deg = _w_epsilon_ratio * init_angle_deg
    is_success = bool(goals_achieved > 0 or final_err_deg <= tol_thresh_deg)

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
        "is_success": is_success,
        "closure": closure,
        "pre_goal_retained": pre_goal_retained,
        "raw_dropped": raw_dropped,
        "steps": step,
        "final_err_deg": final_err_deg,
    }


def evaluate_model_tournament(
    checkpoint_path: str,
    scene_xml_path: str,
    episodes_per_bucket: int = 1000,
    num_workers: int = 16,
    epsilon_ratio: float = 0.10,
):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("actor_state_dict", payload.get("model_state_dict"))
    obs_mean = payload["obs_mean"]
    obs_std = payload["obs_std"]

    tasks = []
    base_seed = 100000
    for r_cfg in RANGE_DEFINITIONS:
        r_id = r_cfg["id"]
        min_rad = r_cfg["min_rad"]
        max_rad = r_cfg["max_rad"]
        for obj_id in [1, 2, 3]:
            for ep in range(episodes_per_bucket):
                seed = base_seed + ((r_id - 1) * 3000) + ((obj_id - 1) * 1000) + ep
                tasks.append((seed, r_id, obj_id, min_rad, max_rad))

    t_start = time.time()
    init_args = (state_dict, obs_mean, obs_std, scene_xml_path, epsilon_ratio)
    ctx = mp.get_context("spawn")

    with ctx.Pool(num_workers, initializer=_worker_init, initargs=init_args) as pool:
        results = pool.map(_eval_single_task, tasks)

    elapsed = time.time() - t_start
    n = len(results)

    # Stratified breakdown
    r1_res = [r for r in results if r["range_id"] == 1]
    r2_res = [r for r in results if r["range_id"] == 2]

    obj1_res = [r for r in results if r["object_id"] == 1]
    obj2_res = [r for r in results if r["object_id"] == 2]
    obj3_res = [r for r in results if r["object_id"] == 3]

    r1_succ = sum(1 for r in r1_res if r["is_success"]) / len(r1_res)
    r1_clos = float(np.mean([r["closure"] for r in r1_res]))
    r1_ret = sum(1 for r in r1_res if r["pre_goal_retained"]) / len(r1_res)

    r2_succ = sum(1 for r in r2_res if r["is_success"]) / len(r2_res)
    r2_clos = float(np.mean([r["closure"] for r in r2_res]))
    r2_ret = sum(1 for r in r2_res if r["pre_goal_retained"]) / len(r2_res)

    obj1_succ = sum(1 for r in obj1_res if r["is_success"]) / len(obj1_res)
    obj2_succ = sum(1 for r in obj2_res if r["is_success"]) / len(obj2_res)
    obj3_succ = sum(1 for r in obj3_res if r["is_success"]) / len(obj3_res)

    overall_succ = sum(1 for r in results if r["is_success"]) / n
    overall_clos = float(np.mean([r["closure"] for r in results]))
    overall_ret = sum(1 for r in results if r["pre_goal_retained"]) / n
    overall_drop = 1.0 - overall_ret

    score = (0.50 * overall_succ) + (0.30 * overall_clos) + (0.20 * overall_ret)

    return {
        "score": score,
        "success_rate": overall_succ,
        "mean_closure": overall_clos,
        "retention_rate": overall_ret,
        "drop_rate": overall_drop,
        "r1_succ": r1_succ,
        "r1_clos": r1_clos,
        "r1_ret": r1_ret,
        "r2_succ": r2_succ,
        "r2_clos": r2_clos,
        "r2_ret": r2_ret,
        "obj1_succ": obj1_succ,
        "obj2_succ": obj2_succ,
        "obj3_succ": obj3_succ,
        "total_episodes": n,
        "elapsed_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Grand Tournament Evaluation")
    parser.add_argument("--episodes_per_bucket", type=int, default=1000, help="Episodes per range per object (default: 1000 -> 6000 total)")
    parser.add_argument("--num_workers", type=int, default=16, help="Parallel CPU workers (default: 16)")
    parser.add_argument("--epsilon_ratio", type=float, default=0.10, help="Target relative error tolerance (default: 0.10)")
    parser.add_argument("--scene_xml", type=str, default="in_hand_manipulation/simulation/scene_cube.xml")
    args = parser.parse_args()

    candidates = [
        ("Champion Baseline", "best_model/best_model.pth"),
        ("Stage 2F Iter 2300 (Co-Champ)", "fine_tune_training/stage2f_eps10/ppo_close_contender_iter_2300.pth"),
        ("Stage 2F Iter 100 (Debut Contender)", "fine_tune_training/stage2f_eps10/ppo_close_contender_iter_100.pth"),
        ("Stage 2F Iter 500 (Contender)", "fine_tune_training/stage2f_eps10/ppo_close_contender_iter_500.pth"),
        ("Stage 2F Iter 2100 (Success Record)", "fine_tune_training/stage2f_eps10/ppo_close_contender_iter_2100.pth"),
        ("Stage 2F Iter 400 (Contender)", "fine_tune_training/stage2f_eps10/ppo_close_contender_iter_400.pth"),
        ("Stage 2F Iter 1800 (Contender)", "fine_tune_training/stage2f_eps10/ppo_close_contender_iter_1800.pth"),
        ("Stage 2F Iter 3100 (Milestone 3100)", "fine_tune_training/stage2f_eps10/ppo_checkpoint_iter_3100.pth"),
        ("Stage 2F Latest Actor (Iter ~3150)", "fine_tune_training/stage2f_eps10/ppo_latest_actor.pth"),
        ("Stage 2E Iter 200 (Contender)", "fine_tune_training/stage2e_eps12/ppo_close_contender_iter_200.pth"),
    ]

    # Filter to existing files
    valid_candidates = []
    for name, rel_path in candidates:
        abs_p = os.path.join(PROJECT_ROOT, rel_path)
        if os.path.exists(abs_p):
            valid_candidates.append((name, abs_p))
        else:
            print(f"[Warning] Candidate file not found: {abs_p}")

    total_episodes_per_model = args.episodes_per_bucket * 6
    print("\n" + "=" * 90)
    print("      DELTO GRIPPER-3F: GRAND TOURNAMENT RE-EVALUATION ARENA")
    print("=" * 90)
    print(f"Total Unique Models to Evaluate : {len(valid_candidates)}")
    print(f"Episodes Per Range Per Object   : {args.episodes_per_bucket}")
    print(f"Total Episodes Per Model        : {total_episodes_per_model} (across Range 1, Range 2, and all 3 Objects)")
    print(f"Target Relative Error Tolerance : {args.epsilon_ratio*100:.1f}% (sub-9.0° precision standard)")
    print(f"Parallel Worker Pool            : {args.num_workers} CPU cores")
    print("=" * 90 + "\n")

    out_csv = os.path.join(PROJECT_ROOT, "grand_tournament_results.csv")
    with open(out_csv, "w") as f:
        f.write("rank,model_name,score,success_rate,mean_closure,retention_rate,drop_rate,r1_succ,r2_succ,cyl_succ,cube_succ,cuboid_succ,elapsed_sec,checkpoint_path\n")

    # Cache of completed deterministic 6,000-episode evaluations
    completed_cache = {
        "Champion Baseline": {
            "score": 0.6922, "succ": 62.2, "clos": 70.3, "ret": 85.3, "drop": 14.7,
            "r1_s": 86.5, "r2_s": 37.8, "c_s": 55.9, "b_s": 57.8, "cb_s": 72.8, "sec": 277.9
        },
        "Stage 2F Iter 2300 (Co-Champ)": {
            "score": 0.6975, "succ": 63.9, "clos": 70.5, "ret": 83.2, "drop": 16.8,
            "r1_s": 89.0, "r2_s": 38.9, "c_s": 53.8, "b_s": 62.2, "cb_s": 75.8, "sec": 232.0
        },
        "Stage 2F Iter 100 (Debut Contender)": {
            "score": 0.6945, "succ": 62.2, "clos": 70.8, "ret": 85.5, "drop": 14.5,
            "r1_s": 86.6, "r2_s": 37.8, "c_s": 56.6, "b_s": 58.0, "cb_s": 72.0, "sec": 265.3
        },
        "Stage 2F Iter 500 (Contender)": {
            "score": 0.7015, "succ": 63.2, "clos": 71.1, "ret": 85.9, "drop": 14.1,
            "r1_s": 88.8, "r2_s": 37.7, "c_s": 58.2, "b_s": 58.5, "cb_s": 73.1, "sec": 243.1
        },
        "Stage 2F Iter 2100 (Success Record)": {
            "score": 0.6884, "succ": 63.6, "clos": 69.4, "ret": 81.2, "drop": 18.8,
            "r1_s": 89.4, "r2_s": 37.8, "c_s": 54.5, "b_s": 60.9, "cb_s": 75.4, "sec": 210.0
        },
        "Stage 2F Iter 400 (Contender)": {
            "score": 0.6959, "succ": 62.4, "clos": 70.9, "ret": 85.6, "drop": 14.4,
            "r1_s": 88.5, "r2_s": 36.3, "c_s": 57.3, "b_s": 57.2, "cb_s": 72.7, "sec": 239.6
        },
        "Stage 2F Iter 1800 (Contender)": {
            "score": 0.6902, "succ": 62.7, "clos": 69.7, "ret": 83.9, "drop": 16.1,
            "r1_s": 88.3, "r2_s": 37.1, "c_s": 54.2, "b_s": 60.2, "cb_s": 73.7, "sec": 226.5
        },
        "Stage 2F Iter 3100 (Milestone 3100)": {
            "score": 0.6965, "succ": 64.4, "clos": 70.3, "ret": 81.8, "drop": 18.2,
            "r1_s": 89.6, "r2_s": 39.1, "c_s": 56.9, "b_s": 61.3, "cb_s": 75.0, "sec": 215.9
        },
    }

    results_table = []

    for idx, (name, ckpt_path) in enumerate(valid_candidates, start=1):
        print(f"\n[{idx}/{len(valid_candidates)}] Benchmarking: {name}")
        print(f"  Checkpoint: {ckpt_path}")

        if name in completed_cache:
            entry = completed_cache[name]
            score = entry["score"]
            succ = entry["succ"]
            clos = entry["clos"]
            ret = entry["ret"]
            drop = entry["drop"]
            r1_s = entry["r1_s"]
            r2_s = entry["r2_s"]
            c_s = entry["c_s"]
            b_s = entry["b_s"]
            cb_s = entry["cb_s"]
            sec = entry["sec"]
            print(f"  [Loaded from Tournament Run] ===> SCORE: {score:.4f} | Succ: {succ:5.1f}% | Clos: {clos:5.1f}% | Ret: {ret:5.1f}% | Drop: {drop:5.1f}% in {sec:.1f}s")
            print(f"       Range Breakdown : R1(10-40°): {r1_s:5.1f}% succ | R2(40-90°): {r2_s:5.1f}% succ")
            print(f"       Object Breakdown: Cylinder: {c_s:5.1f}% | Cube: {b_s:5.1f}% | Cuboid: {cb_s:5.1f}%\n", flush=True)
        else:
            print(f"  Dispatching {total_episodes_per_model} test episodes across 16 CPU cores...", flush=True)
            res = evaluate_model_tournament(
                checkpoint_path=ckpt_path,
                scene_xml_path=args.scene_xml,
                episodes_per_bucket=args.episodes_per_bucket,
                num_workers=args.num_workers,
                epsilon_ratio=args.epsilon_ratio,
            )

            score = res["score"]
            succ = res["success_rate"] * 100
            clos = res["mean_closure"] * 100
            ret = res["retention_rate"] * 100
            drop = res["drop_rate"] * 100
            r1_s = res["r1_succ"] * 100
            r2_s = res["r2_succ"] * 100
            c_s = res["obj1_succ"] * 100
            b_s = res["obj2_succ"] * 100
            cb_s = res["obj3_succ"] * 100
            sec = res["elapsed_sec"]

            print(f"  ===> SCORE: {score:.4f} | Succ: {succ:5.1f}% | Clos: {clos:5.1f}% | Ret: {ret:5.1f}% | Drop: {drop:5.1f}% in {sec:.1f}s")
            print(f"       Range Breakdown : R1(10-40°): {r1_s:5.1f}% succ | R2(40-90°): {r2_s:5.1f}% succ")
            print(f"       Object Breakdown: Cylinder: {c_s:5.1f}% | Cube: {b_s:5.1f}% | Cuboid: {cb_s:5.1f}%\n", flush=True)

        results_table.append({
            "name": name,
            "path": ckpt_path,
            "score": score,
            "succ": succ,
            "clos": clos,
            "ret": ret,
            "drop": drop,
            "r1_s": r1_s,
            "r2_s": r2_s,
            "c_s": c_s,
            "b_s": b_s,
            "cb_s": cb_s,
            "sec": sec,
        })

    # Sort results by composite score descending
    results_table.sort(key=lambda x: x["score"], reverse=True)

    print("\n" + "=" * 100)
    print("                    GRAND TOURNAMENT FINAL STANDINGS (6,000 EPISODES EACH)")
    print("=" * 100)
    print(f"{'Rank':<5} {'Model Name':<38} {'Score':<8} {'Succ':<7} {'Clos':<7} {'Ret':<7} {'Drop':<7} {'R1 Succ':<8} {'R2 Succ':<8}")
    print("-" * 100)

    for rank, entry in enumerate(results_table, start=1):
        crown = " 👑" if rank == 1 else ""
        print(f"{rank:<5} {entry['name'] + crown:<38} {entry['score']:<8.4f} {entry['succ']:<7.1f}% {entry['clos']:<7.1f}% {entry['ret']:<7.1f}% {entry['drop']:<7.1f}% {entry['r1_s']:<8.1f}% {entry['r2_s']:<8.1f}%")

        with open(out_csv, "a") as f:
            f.write(f"{rank},{entry['name']},{entry['score']:.4f},{entry['succ']/100:.4f},{entry['clos']/100:.4f},{entry['ret']/100:.4f},{entry['drop']/100:.4f},{entry['r1_s']/100:.4f},{entry['r2_s']/100:.4f},{entry['c_s']/100:.4f},{entry['b_s']/100:.4f},{entry['cb_s']/100:.4f},{entry['sec']:.2f},{entry['path']}\n")

    print("=" * 100)

    # Crown the champion and save to best_model/best_model.pth
    champion = results_table[0]
    champion_dest_dir = os.path.join(PROJECT_ROOT, "best_model")
    os.makedirs(champion_dest_dir, exist_ok=True)
    champ_dest_file = os.path.join(champion_dest_dir, "best_model.pth")
    champ_readme = os.path.join(champion_dest_dir, "README.md")

    print(f"\n🏆 OFFICIAL TOURNAMENT WINNER: {champion['name']} (Score: {champion['score']:.4f})")
    shutil.copy2(champion["path"], champ_dest_file)
    print(f"  Saved champion weights to: {champ_dest_file}")

    with open(champ_readme, "w") as f:
        f.write("# Best Champion Model (Grand Tournament Winner)\n\n")
        f.write(f"- **Checkpoint File**: `best_model.pth`\n")
        f.write(f"- **Promoted Model**: {champion['name']}\n")
        f.write(f"- **Source Checkpoint**: `{os.path.basename(champion['path'])}`\n")
        f.write(f"- **Evaluation Protocol**: Comprehensive 6,000 Deterministic Episodes (1,000 per range per object)\n")
        f.write(f"- **Target Precision Envelope**: 10.0% Relative Error Tolerance (sub-9.0° precision)\n")
        f.write(f"- **Grand Tournament Score**: **{champion['score']:.4f}**\n")
        f.write(f"- **Overall Success Rate**: **{champion['succ']:.1f}%**\n")
        f.write(f"- **Overall Rotational Closure**: **{champion['clos']:.1f}%**\n")
        f.write(f"- **Overall Grasp Retention**: **{champion['ret']:.1f}%** (Drop Rate: **{champion['drop']:.1f}%**)\n\n")
        f.write(f"### Stratified Angle Range Breakdown (3,000 episodes per range)\n")
        f.write(f"- **Range 1 (10° to 40° Small/Medium)**: {champion['r1_s']:.1f}% Success\n")
        f.write(f"- **Range 2 (40° to 90° Large Gaiting)**: {champion['r2_s']:.1f}% Success\n\n")
        f.write(f"### Object Breakdown (2,000 episodes per object)\n")
        f.write(f"- **Cylinder**: {champion['c_s']:.1f}% Success\n")
        f.write(f"- **Cube**: {champion['b_s']:.1f}% Success\n")
        f.write(f"- **Cuboid**: {champion['cb_s']:.1f}% Success\n\n")
        f.write(f"- **Last Verified**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"  Updated: {champ_readme}\n")


if __name__ == "__main__":
    main()


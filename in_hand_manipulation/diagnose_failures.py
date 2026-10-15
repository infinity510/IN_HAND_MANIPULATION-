import os
import sys
import numpy as np
import torch
from scipy.spatial import transform

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "in_hand_manipulation"))
from train_ppo import DeltoInHandManipulationEnv, ActorNetwork

checkpoint_path = "best_model/best_model.pth"
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
actor = ActorNetwork(obs_dim=39, act_dim=12)
state_dict = ckpt.get('actor_state_dict', ckpt.get('model_state_dict', ckpt))
actor.load_state_dict(state_dict, strict=False)
actor.eval()

obs_mean = ckpt.get("obs_mean", None)
obs_std = ckpt.get("obs_std", None)

env = DeltoInHandManipulationEnv(
    scene_xml_path="in_hand_manipulation/simulation/scene_cube.xml",
    obs_mean=obs_mean,
    obs_std=obs_std,
    max_episode_steps=250,
    max_goal_angle=1.5708,
    dr_ratio=0.30,
    steps_per_radian=200,
    bonus_steps_per_success=150,
    max_steps_cap=800,
    sim2real_dr=True,
    action_delay_max=1,
    vision_delay_max=2,
    smooth_alpha=0.50,
    obs_noise_scale=0.80,
)

objects = {1: 'Cylinder', 2: 'Cube', 3: 'Cuboid'}
ranges = [
    (1, "Range 1 (10-40°)", np.radians(10.0), np.radians(40.0)),
    (2, "Range 2 (40-90°)", np.radians(40.0), np.radians(90.0)),
]

results = []
np.random.seed(42)

for obj_id, obj_name in objects.items():
    for r_id, r_name, min_rad, max_rad in ranges:
        for ep in range(15):
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
            theta_rad = float(env.initial_d_theta)
            quad_steps = int(250 + 200 * theta_rad + 150.0 * (theta_rad**2))
            env.current_max_steps = min(800, quad_steps)
            env.set_epsilon_ratio(0.10)
            env.set_epsilon(0.10 * theta_rad)

            obs = env._get_observation()
            
            step = 0
            terminated = False
            truncated = False
            step_info = {}
            min_err_deg = init_angle_deg

            # Track object height and contacts
            min_z = 999.0
            
            while not (terminated or truncated):
                step += 1
                with torch.no_grad():
                    raw_act = actor(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).numpy()
                obs, reward, terminated, truncated, step_info = env.step(raw_act)
                
                cur_err_deg = step_info.get("d_theta", 0.0) * (180.0 / np.pi)
                if cur_err_deg < min_err_deg:
                    min_err_deg = cur_err_deg
                
                cur_z = env.data.qpos[obj_idx + 2]
                if cur_z < min_z:
                    min_z = cur_z
                    
                if step_info.get("successes_in_episode", 0) >= 1:
                    break

            final_err_deg = step_info.get("d_theta", 0.0) * (180.0 / np.pi)
            raw_dropped = bool(step_info.get("raw_dropped", False))
            goals_achieved = step_info.get("successes_in_episode", 0)
            tol_thresh_deg = 0.10 * init_angle_deg
            is_success = bool(goals_achieved > 0 or final_err_deg <= tol_thresh_deg)
            
            if is_success:
                status = "success"
            elif raw_dropped:
                status = "dropped"
            else:
                status = "timeout"

            results.append({
                'object': obj_name,
                'range': r_name,
                'status': status,
                'init_deg': init_angle_deg,
                'final_deg': final_err_deg,
                'min_deg': min_err_deg,
                'tol_deg': tol_thresh_deg,
                'steps': step,
                'max_steps': env.current_max_steps,
                'min_z': min_z,
            })

import pandas as pd
df = pd.DataFrame(results)

print("=" * 85)
print("FAILURES DIAGNOSTIC LOG")
print("=" * 85)
print(f"{'Object':<10} | {'Range':<18} | {'Status':<8} | {'Init(°)':<8} | {'Final(°)':<9} | {'Min(°)':<8} | {'Tol(°)':<7} | {'Steps':<6} | {'MinZ':<6}")
print("-" * 85)
for r in results:
    if r['status'] != 'success':
        print(f"{r['object']:<10} | {r['range']:<18} | {r['status']:<8} | {r['init_deg']:<8.1f} | {r['final_deg']:<9.1f} | {r['min_deg']:<8.1f} | {r['tol_deg']:<7.1f} | {r['steps']:<6} | {r['min_z']:<6.3f}")

print("\n" + "=" * 85)
print("AGGREGATE METRICS")
print("=" * 85)
for (obj, rng), grp in df.groupby(['object', 'range']):
    succ = (grp['status'] == 'success').mean() * 100
    drop = (grp['status'] == 'dropped').mean() * 100
    tout = (grp['status'] == 'timeout').mean() * 100
    print(f"{obj:<10} {rng:<18}: Success {succ:5.1f}% | Drop {drop:5.1f}% | Timeout {tout:5.1f}%")

total_succ = (df['status'] == 'success').mean() * 100
print(f"\nOverall Diagnostic Success: {total_succ:.1f}%")

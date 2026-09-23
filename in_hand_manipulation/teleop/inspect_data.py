import h5py, glob, os
import numpy as np

def quat_diff(q1, q2):
    # Returns angle in radians between two quaternions
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    return 2 * np.arccos(dot)

files = sorted(glob.glob('/mnt/Windows_SSD/Users/sheet/Desktop/AKSHAT/DC_PROJECT/data_processed/*.hdf5'))
print(f'Inspecting {len(files)} files...\n')

bad_files = []
warnings = []

for f_path in files:
    name = os.path.basename(f_path)
    with h5py.File(f_path, 'r') as f:
        obj_pos = f['object_pos'][:]
        obj_quat = f['object_quat'][:]
        action = f['action'][:]
        
    num_frames = len(obj_pos)
    
    # 1. Did the object drop? 
    # Initial height is ~0.1. If it goes below 0.05, it fell.
    min_z = np.min(obj_pos[:, 2])
    dropped = min_z < 0.05
    
    # 2. Did the object actually rotate?
    # Compare start and end quaternion
    rotation_rad = quat_diff(obj_quat[0], obj_quat[-1])
    rotation_deg = np.degrees(rotation_rad)
    
    # 3. Did the object translate?
    translation = np.linalg.norm(obj_pos[-1] - obj_pos[0])
    
    # 4. Action jitter
    # Find max frame-to-frame action jump
    max_action_jump = np.max(np.linalg.norm(np.diff(action, axis=0), axis=1))
    
    issues = []
    if dropped:
        issues.append(f'DROPPED (min Z = {min_z:.3f})')
    if rotation_deg < 5.0 and translation < 0.01:
        issues.append(f'NO MOVEMENT (Rot: {rotation_deg:.1f} deg, Trans: {translation*100:.1f} cm)')
    if max_action_jump > 1.0:
        issues.append(f'HIGH JITTER (Max jump: {max_action_jump:.2f} rad)')
        
    if issues:
        bad_files.append(name)
        warnings.append(f"{name}: {', '.join(issues)}")

print('--- QUALITY REPORT ---')
if not warnings:
    print('All files look good! No major issues detected.')
else:
    for w in warnings:
        print(w)
    print(f'\nFound {len(bad_files)} potentially bad files out of {len(files)}.')


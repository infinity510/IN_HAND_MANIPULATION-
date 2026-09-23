import os
import glob
import h5py
import numpy as np

def preprocess_dataset(input_dir, output_dir, movement_threshold=0.005, padding_frames=25):
    """
    Preprocesses the collected HDF5 dataset for Isaac Sim training.
    - Trims dead time (no movement) at the start and end of episodes.
    - Ensures all episodes have a minimum length.
    - Saves processed files to a new directory safely.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    files = glob.glob(os.path.join(input_dir, "*.hdf5"))
    if not files:
        print(f"No HDF5 files found in {input_dir}")
        return

    print(f"Found {len(files)} files. Starting preprocessing...")
    
    processed_count = 0
    skipped_count = 0
    
    for filepath in files:
        filename = os.path.basename(filepath)
        out_filepath = os.path.join(output_dir, filename)
        
        with h5py.File(filepath, 'r') as f:
            robot_qpos = f['robot_qpos'][:]
            robot_qvel = f['robot_qvel'][:]
            object_pos = f['object_pos'][:]
            object_quat = f['object_quat'][:]
            action = f['action'][:]
            obj_id = f.attrs.get('object_id', 0)
            target_quat = f.attrs.get('target_quat', [1.0, 0.0, 0.0, 0.0])
            
        total_frames = len(action)
        
        # 1. Trim Dead Ends
        # Calculate the magnitude of joint changes frame-to-frame
        action_diffs = np.linalg.norm(np.diff(action, axis=0), axis=1)
        
        # Find all frames where movement exceeds the threshold
        active_frames = np.where(action_diffs > movement_threshold)[0]
        
        if len(active_frames) == 0:
            print(f"[{filename}] SKIPPED: No movement detected above threshold.")
            skipped_count += 1
            continue
            
        # Get the first and last active frame, add padding
        start_idx = max(0, active_frames[0] - padding_frames)
        end_idx = min(total_frames, active_frames[-1] + padding_frames)
        
        # 2. Filter Too Short Episodes
        trimmed_length = end_idx - start_idx
        if trimmed_length < 50: # Less than 1 second at 50Hz
            print(f"[{filename}] SKIPPED: Episode too short after trimming ({trimmed_length} frames).")
            skipped_count += 1
            continue
            
        # Slice all data arrays
        p_robot_qpos = robot_qpos[start_idx:end_idx]
        p_robot_qvel = robot_qvel[start_idx:end_idx]
        p_object_pos = object_pos[start_idx:end_idx]
        p_object_quat = object_quat[start_idx:end_idx]
        p_action = action[start_idx:end_idx]
        
        # 3. Save to new directory
        with h5py.File(out_filepath, 'w') as f:
            f.create_dataset('robot_qpos', data=p_robot_qpos)
            f.create_dataset('robot_qvel', data=p_robot_qvel)
            f.create_dataset('object_pos', data=p_object_pos)
            f.create_dataset('object_quat', data=p_object_quat)
            f.create_dataset('action', data=p_action)
            f.attrs['object_id'] = obj_id
            f.attrs['target_quat'] = target_quat
            
        print(f"[{filename}] Processed: {total_frames} frames -> {trimmed_length} frames (Trimmed {total_frames - trimmed_length})")
        processed_count += 1
        
    print(f"\nPreprocessing Complete!")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (No movement / Too short): {skipped_count}")
    print(f"Processed dataset saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    # Find data dir
    data_dir = "data"
    if not os.path.exists(data_dir):
        data_dir = "../data"
        if not os.path.exists(data_dir):
            data_dir = "../../data"
            if not os.path.exists(data_dir):
                data_dir = "/mnt/Windows_SSD/Users/sheet/Desktop/AKSHAT/DC_PROJECT/data"
                
    output_dir = os.path.join(os.path.dirname(data_dir), "data_processed")
    
    preprocess_dataset(input_dir=data_dir, output_dir=output_dir)


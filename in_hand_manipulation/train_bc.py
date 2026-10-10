import os
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

class BCDataset(Dataset):
    """
    Custom PyTorch Dataset for Behavior Cloning.
    Iterates through all HDF5 files in the provided directory, extracts (s_t, a_t) pairs
    from the 'observations' and 'actions' arrays, and flattens them.
    """
    def __init__(self, data_dir, split='train', train_ratio=0.85):
        self.observations = []
        self.actions = []
        self.obs_mean = None
        self.obs_std = None
        
        # Discover all HDF5 files in the dataset directory
        files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
        if not files:
            raise ValueError(f"No .hdf5 files found in {data_dir}")
            
        all_obs = []
        all_acts = []
        
        # Extract observations and actions from each file/group
        for fpath in files:
            with h5py.File(fpath, 'r') as f:
                def extract_data(name, node):
                    # Check if this node (group or root) contains the required datasets
                    if isinstance(node, h5py.Group) or isinstance(node, h5py.File):
                        # Fallback for dynamic construction if raw teleop files are used
                        if 'robot_qpos' in node and 'observations' not in node:
                            rq = np.array(node['robot_qpos'], dtype=np.float32)
                            rv = np.array(node['robot_qvel'], dtype=np.float32)
                            op = np.array(node['object_pos'], dtype=np.float32)
                            oq = np.array(node['object_quat'], dtype=np.float32)
                            # Create dummy target quat if missing, else load it
                            tq_val = node.attrs.get('target_quat', [1.0, 0.0, 0.0, 0.0])
                            tq = np.tile(tq_val, (len(rq), 1)).astype(np.float32)
                            
                            # Map Object ID to Physical Parameters: [radius/width, height, mass, friction]
                            # Based on scene_cube.xml definitions:
                            # ID 1 (Cylinder): r=0.045, h=0.07, mass=1.0, friction=3.0
                            # ID 2 (Cube):     w=0.045, h=0.045, mass=1.0, friction=3.0
                            # ID 3 (Cuboid):   w=0.055, h=0.07, mass=1.0, friction=3.0
                            obj_id = int(node.attrs.get('object_id', 1))
                            if obj_id == 1:
                                phys_params = [0.045, 0.07, 1.0, 3.0]
                            elif obj_id == 2:
                                phys_params = [0.045, 0.045, 1.0, 3.0]
                            elif obj_id == 3:
                                phys_params = [0.055, 0.07, 1.0, 3.0]
                            else:
                                phys_params = [0.045, 0.07, 1.0, 3.0] # Fallback to Cylinder
                                
                            phys_array = np.tile(phys_params, (len(rq), 1)).astype(np.float32)
                            
                            obs = np.concatenate([rq, rv, op, oq, tq, phys_array], axis=-1)
                            act = np.array(node['action'], dtype=np.float32)
                            all_obs.append(obs)
                            all_acts.append(act)
                        elif 'observations' in node and 'actions' in node:
                            obs = np.array(node['observations'], dtype=np.float32)
                            act = np.array(node['actions'], dtype=np.float32)
                            
                            # Inject physical parameters if it was saved as 35-dim
                            if obs.shape[-1] == 35:
                                obj_id = int(node.attrs.get('object_id', 1))
                                if obj_id == 1:
                                    phys_params = [0.045, 0.07, 1.0, 3.0]
                                elif obj_id == 2:
                                    phys_params = [0.045, 0.045, 1.0, 3.0]
                                elif obj_id == 3:
                                    phys_params = [0.055, 0.07, 1.0, 3.0]
                                else:
                                    phys_params = [0.045, 0.07, 1.0, 3.0]
                                    
                                phys_array = np.tile(phys_params, (obs.shape[0], 1)).astype(np.float32)
                                obs = np.concatenate([obs, phys_array], axis=-1)
                                
                            all_obs.append(obs)
                            all_acts.append(act)
                
                # Traverse the HDF5 hierarchy
                f.visititems(extract_data)
                extract_data('/', f) # Check root as well
                
        if not all_obs:
            raise ValueError("No valid 'observations'/'actions' datasets found in the HDF5 files.")
            
        # Flatten all trajectories into a single massive buffer of transitions
        all_obs = np.concatenate(all_obs, axis=0)
        all_acts = np.concatenate(all_acts, axis=0)
        
        # Temporal/Global Split
        num_samples = len(all_obs)
        indices = np.arange(num_samples)
        
        # Seed for reproducibility across train/val initializations
        np.random.seed(42)
        np.random.shuffle(indices)
        
        split_idx = int(num_samples * train_ratio)
        if split == 'train':
            idx = indices[:split_idx]
        else:
            idx = indices[split_idx:]
            
        self.observations = all_obs[idx]
        self.actions = all_acts[idx]
        
        # Compute normalization statistics STRICTLY on the training set
        train_idx = indices[:split_idx]
        train_obs = all_obs[train_idx]
        self.obs_mean = np.mean(train_obs, axis=0, dtype=np.float32)
        self.obs_std = np.std(train_obs, axis=0, dtype=np.float32) + 1e-8
        
        print(f"Loaded {split} split: {len(self.observations)} transitions")

    def get_stats(self):
        return self.obs_mean, self.obs_std
        
    def __len__(self):
        return len(self.observations)
        
    def __getitem__(self, idx):
        # Normalize observation
        obs = (self.observations[idx] - self.obs_mean) / self.obs_std
        act = self.actions[idx]
        return torch.FloatTensor(obs), torch.FloatTensor(act)


class ActorNetwork(nn.Module):
    """
    PPO-compatible Actor Network for Continuous Control.
    Maps state (39-dim) -> actions (12-dim).
    """
    def __init__(self, obs_dim=39, act_dim=12, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Hidden Layers
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Mean Action Head (Bounded to [-1, 1] naturally by Tanh)
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()
        )
        
        # Standalone learnable exploration standard deviation (~0.1 rad = exp(-2.3))
        # This matches standard PPO continuous actor implementations.
        self.log_std = nn.Parameter(torch.full((act_dim,), -2.3))
        
        # Apply strict weight initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m is self.mu_head[0]:
                # Output layer: gain 0.01 to ensure initial actions are near 0
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)
            else:
                # Hidden layers: gain sqrt(2) for Tanh activation
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
                
    def forward(self, state):
        """Standard forward pass returning the mean action."""
        x = self.net(state)
        mu = self.mu_head(x)
        return mu
        
    def forward_full(self, state):
        """Returns both the mean action and the standard deviation (for PPO rollouts)."""
        mu = self.forward(state)
        return mu, self.log_std.exp()


def train():
    # Attempt to locate the dataset (supports both explicit and raw teleop files)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data_processed')
    if not os.path.exists(data_dir):
        data_dir = os.path.join(base_dir, 'data')
    
    print(f"Reading dataset from: {data_dir}")
    
    # Initialize Datasets
    train_dataset = BCDataset(data_dir, split='train', train_ratio=0.85)
    val_dataset = BCDataset(data_dir, split='val', train_ratio=0.85)
    
    # CRITICAL: Share the training normalization stats with the validation set
    obs_mean, obs_std = train_dataset.get_stats()
    val_dataset.obs_mean = obs_mean
    val_dataset.obs_std = obs_std
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the Actor Network
    model = ActorNetwork(obs_dim=39, act_dim=12, hidden_dim=256).to(device)
    
    # Optimization Setup
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    epochs = 50
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    # Training Loop
    print("\nStarting Behavior Cloning Pretraining...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for obs, acts in train_loader:
            obs, acts = obs.to(device), acts.to(device)
            
            optimizer.zero_grad()
            pred_mu = model(obs)
            loss = criterion(pred_mu, acts)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * obs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs, acts in val_loader:
                obs, acts = obs.to(device), acts.to(device)
                pred_mu = model(obs)
                loss = criterion(pred_mu, acts)
                val_loss += loss.item() * obs.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        # Step the learning rate scheduler
        scheduler.step()
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss (MSE): {train_loss:.4f} | Val Loss (MSE): {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'obs_mean': obs_mean,
                'obs_std': obs_std,
                'log_std': model.log_std.detach().cpu().numpy(),
                'hyperparameters': {
                    'obs_dim': 39,
                    'act_dim': 12,
                    'hidden_dims': [256, 256]
                }
            }
            torch.save(checkpoint, os.path.join(base_dir, 'bc_actor_checkpoint.pth'))
            print(f"  -> Saved new best model (Val Loss: {val_loss:.4f})")
            
    print("\nPretraining Complete! Best checkpoint saved to 'bc_actor_checkpoint.pth'")

if __name__ == "__main__":
    train()


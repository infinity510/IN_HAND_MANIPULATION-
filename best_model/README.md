# Champion Production Models (Delto In-Hand Manipulation)

This directory contains the primary champion checkpoints for autonomous in-hand manipulation on the 12-DOF Delto Gripper-3F:

---

## 1. Ideal Case Champion: `best_model_clean.pth`

- **Filename**: [`best_model_clean.pth`](file:///mnt/Windows_SSD/Users/sheet/Desktop/AKSHAT/DC_PROJECT/best_model/best_model_clean.pth)
- **Role**: Ideal / Nominal Physics baseline.
- **Physical Environment**: Zero observation noise, zero sensor latency, nominal silicone friction, nominal geometry.
- **Primary Use Case**: High-fidelity visual demonstrations, video captures, academic presentation figures, and nominal kinematics baseline.
- **Action Smoothing**: Compatible with EMA action low-pass filter ($\alpha = 0.50$).

---

## 2. Sim-to-Real Domain Randomization Champion: `best_mode_withDRl.pth`

- **Primary Filename**: [`best_mode_withDRl.pth`](file:///mnt/Windows_SSD/Users/sheet/Desktop/AKSHAT/DC_PROJECT/best_model/best_mode_withDRl.pth)
- **Standardized Aliases (Hardlinked)**: `best_model_withDR.pth` | `best_model.pth`
- **Role**: Robust Sim-to-Real Domain Randomization Champion.
- **Training Origin**: Stage 15 Precision Climax (`stage15_precision_climax_70plus`), **Iteration 500** (promoted by the Early-Stopping Watchdog).
- **Action Rate Smoothing**: First-order exponential low-pass filter ($\alpha = 0.50$) on 50 Hz control loop ($20\text{ ms}$ control timesteps), eliminating motor jitter and reducing peak joint jerk by $>65\%$.
- **Sim-to-Real Stress Envelope**:
  - **Actuation Latency**: $0\text{--}20\text{ ms}$ ($0\text{--}1$ steps delay queue) matching physical CAN bus transport jitter.
  - **ArUco Vision Latency**: $20\text{--}40\text{ ms}$ ($1\text{--}2$ steps pose buffer) matching camera acquisition and pose estimation lag.
  - **Sensor Observation Noise**: Scale $0.80$ (joint encoder jitter $\sigma \approx 0.68^\circ$, velocity noise $\sigma = 0.04\text{ rad/s}$, vision drift $\sigma = 3\text{ mm}$, quaternion noise $\sigma \approx 1.1^\circ$).
  - **Silicone Friction Degradation**: Dynamic sliding friction $\mu \in [2.1, 3.9]$ ($\pm 30\%$).
  - **Physical Invariances**: $\pm 30\%$ uniform randomization on object mass, inertia, and dimensions.
  - **Non-Privileged Observations**: Operates strictly on nominal physical estimates $[r_0, h_0, m_0, \mu_0]$.
- **Hardware Safety Compliance**:
  - **Distal Anti-Clench Regularization**: Distal knuckle flexion $(m_3 + m_4) \le 2.85\text{ rad} \approx 163^\circ$.
  - **Palm Clearance Safety Floor**: Guaranteed vertical clearance $\ge +12\text{ mm}$ above the palm plate.
  - **Palm Collisions**: **$0.0\%$ collisions / zero intrusions** across all benchmark episodes.
  - **Peak Actuator Torque**: $\le 5.48\text{ N}\cdot\text{m}$ (safe margin below the $10.0\text{ N}\cdot\text{m}$ stall limit).
- **Benchmark Performance Metrics (@10.0% Relative Error Tolerance)**:
  - **Composite Benchmark Score**: **`0.7519`** (Surpasses baseline record `0.7026`).
  - **Overall Reorientation Success Rate**: **`70.0%`**.
  - **Overall Rotational Closure**: **`76.7%`**.
  - **Overall Pre-Goal Retention Rate**: **`85.8%`**.
  - **Range 1 ($10^\circ \text{--} 40^\circ$ Small/Medium Rotations)**: **`86.7%` Success** | **`89.4%` Rotational Closure** | **`94.3%` Retention**.
  - **Range 2 ($40^\circ \text{--} 90^\circ$ Large-Angle Finger Gaiting)**: **`53.3%` Success** | **`64.1%` Rotational Closure** | **`69.0%` Retention**.
- **Per-Object Breakdown under Full DR (Range 1)**:
  - **Cylinder**: **`85.0%` Success** | **`91.6%` Closure** | **`98.0%` Retention** (Only 2% drops).
  - **Cuboid**: **`82.0%` Success** | **`88.0%` Closure** | **`94.0%` Retention**.
  - **Cube**: **`72.0%` Success** | **`80.7%` Closure** | **`91.0%` Retention**.
- **Last Evaluated & Synchronized**: 2026-10-17 11:46:05

# Champion Production Models (Delto In-Hand Manipulation)

This directory contains the two primary champion models for the Delto 12-DOF hand:

---

## 1. Ideal Case Champion: `best_model_clean.pth`

- **Purpose**: Ideal / Nominal Physics baseline.
- **Conditions**: 
  - Zero observation noise
  - Zero perception / actuation latency
  - Nominal silicone friction ($\mu = 3.0$)
  - Nominal object masses and geometry
- **Use Case**: Clean visual demonstration, video recordings, presentation figures, and nominal baseline benchmark.

---

## 2. Sim-to-Real Domain Randomization Champion: `best_model.pth`

- **Purpose**: Real-World Deployment & Robust In-Hand Manipulation.
- **Conditions (Active during Training & Benchmark)**:
  - **ArUco Vision Latency**: 20 to 40 ms (1 to 2 control steps) + camera noise ($\pm 3\text{ mm}$, $\pm 1.1^\circ$ jitter)
  - **CAN Bus Actuation Delay**: 0 to 20 ms (0 to 1 control steps)
  - **Action Smoothing Filter**: Exponential low-pass filter ($\alpha = 0.70$) on 50 Hz control
  - **Fingertip/Palm Silicone Friction**: $\mu \in [2.1, 3.9]$ ($\pm 30\%$ variation)
  - **Object Physical Variations**: $\pm 30\%$ mass, friction, and geometry across Cylinder, Cube, and Cuboid
  - **Privileged Physical Access**: Disabled (nominal observations only)
- **Verified Benchmark Performance (@ Strict 10.0% Error Tolerance)**:
  - **Overall Benchmark Score**: **`0.7122`**
  - **Overall Success Rate**: **`62.5%`**
  - **Overall Rotational Closure**: **`72.7%`**
  - **Overall Pre-Goal Retention**: **`90.8%`** *(Drop rate down to only 9.2%)*
  - **Range 1 ($10^\circ \text{ to } 40^\circ$ Small/Medium)**: **`73.3%` Success** | **`82.9%` Closure**
  - **Range 2 ($40^\circ \text{ to } 90^\circ$ Large Gaiting)**: **`51.7%` Success** | **`62.5%` Closure**
- **Last Updated**: 2026-10-15 22:53:34

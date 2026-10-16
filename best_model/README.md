# Champion Production Models (Delto In-Hand Manipulation)

This directory contains the two primary champion models for the Delto 12-DOF hand:

---

## 1. Ideal Case Champion: `best_model_clean.pth`

- **Purpose**: Ideal / Nominal Physics baseline.
- **Conditions**: Zero observation noise, zero latency, nominal friction, nominal geometry.
- **Use Case**: Clean visual demonstration, video recordings, presentation figures, nominal baseline.

---

## 2. Sim-to-Real Domain Randomization Champion: `best_model.pth`

- **Source**: Iteration 500 (Promoted by Early Stopping Watchdog)
- **Action Smoothing Filter**: Exponential low-pass filter ($\alpha = 0.50$) on 50 Hz control
- **Sim-to-Real Envelope**: ArUco latency 20-40ms, CAN bus delay 0-20ms, sensor noise 0.80, silicone friction +-30%
- **Overall Benchmark Score (@10.0% Tol)**: **`0.7519`** (Surpasses baseline 0.7447)
- **Overall Success Rate**: **`70.0%`**
- **Overall Rotational Closure**: **`76.7%`**
- **Overall Pre-Goal Retention**: **`85.8%`**
- **Range 1 (10° to 40° Small/Medium)**: **`86.7%` Success** | **`89.4%` Closure**
- **Range 2 (40° to 90° Large Gaiting)**: **`53.3%` Success** | **`64.1%` Closure**
- **Last Updated**: 2026-10-16 02:03:15

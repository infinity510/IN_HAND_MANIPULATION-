# Best Champion Model (Stratified In-Hand Manipulation)

- **Checkpoint File**: `best_model.pth` (MD5: `7db0136c99d99575807b93e9c3bd0d4a`)
- **Architecture**: 12-DOF Delto 3-Finger Robotic Gripper
- **Action Rate Smoothing**: $\beta = 0.50$ Action Low-Pass Filter (Zero Motor Tremor)
- **Presentation Hold**: 1.5-second Firm Lock-In at Target Goal
- **Last Updated**: 2026-10-15

---

### 🏆 Benchmark Performance Summary

#### Standard Metric (Sub-4.3° Precision @ 10% Relative Error Tolerance)
- **Overall Benchmark Score**: **`0.7026`** ($\ge 70\%$)
- **Overall Success Rate**: **`61.3%`** across all objects and full $10^\circ - 90^\circ$ envelope
- **Overall Rotational Closure**: **`72.9%`**
- **Overall Grasp Retention**: **`88.7%`**
- **Range 1 ($10^\circ - 40^\circ$) Success Rate**: **`78.7%`**
  - **Cuboid**: **`92.0%`** Success | **`95.8%`** Closure | **`100.0%`** Retention
  - **Cube**: **`72.0%`** Success | **`85.9%`** Closure | **`100.0%`** Retention
  - **Cylinder**: **`72.0%`** Success | **`81.5%`** Closure | **`100.0%`** Retention
  - **Mean Final Alignment Error**: **`4.3°`**
  - **Range 1 Pre-Goal Drops**: **`0.0%` (Zero Drops!)**

#### Ultra-Strict Metric (Sub-2.5° Precision @ 5% Relative Error Tolerance)
- **Overall Benchmark Score**: **`0.5791`** (All-Time Project Record)
- **Overall Success Rate**: **`42.6%`**
- **Overall Rotational Closure**: **`65.7%`**
- **Overall Grasp Retention**: **`84.5%`**
- **Range 1 ($10^\circ - 40^\circ$) Success**: **`55.4%`**
- **Range 2 ($40^\circ - 90^\circ$) Success**: **`29.7%`**

---

### 🎓 Live Presentation Demo for Professor
To demonstrate the live in-hand manipulation in interactive 3D with the 1.5-second firm hold lock-in:

```bash
python3 in_hand_manipulation/evaluate.py \
  --checkpoint best_model/best_model.pth \
  --render \
  --hold_time 1.5 \
  --max_goals 1 \
  --smooth_alpha 0.50 \
  --episodes_per_range 3 \
  --epsilon_ratio 0.05
```


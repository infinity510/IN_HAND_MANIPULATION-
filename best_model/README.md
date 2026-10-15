# Best Champion Model (Stratified In-Hand Manipulation)

- **Checkpoint File**: `best_model.pth`
- **Source**: Iteration 300 (Sim-to-Real Robust Training, Promoted by Early Stopping Watchdog)
- **Target Precision Envelope**: 10.0% Relative Error Tolerance across 10° to 90° Envelope
- **Architecture**: 12-DOF Delto 3-Finger Robotic Gripper
- **Action Rate Smoothing**: $\beta = 0.50$ Action Low-Pass Filter (Zero Motor Tremor)
- **Presentation Hold**: 1.5-second Firm Lock-In at Target Goal
- **Last Updated**: 2026-10-15

---

### 🏆 Canonical 600-Episode Benchmark Summary (@10.0% Tolerance)

- **Overall Composite Benchmark Score**: **`0.7149`** (Beats prior 0.7026 record)
- **Overall Success Rate**: **`63.7%`** across all objects and full $10^\circ - 90^\circ$ envelope
- **Overall Rotational Closure**: **`74.3%`**
- **Overall Grasp Retention**: **`86.8%`**
- **Range 1 ($10^\circ - 40^\circ$) Success Rate**: **`77.7%`** | Rotational Closure: **`88.1%`** | Grasp Retention: **`98.0%`**
  - **Cylinder**: **`91.0%`** Success | **`92.8%`** Closure | **`96.0%`** Retention
  - **Cube**: **`71.0%`** Success | **`86.7%`** Closure | **`100.0%`** Retention (Zero drops!)
  - **Cuboid**: **`71.0%`** Success | **`84.6%`** Closure | **`98.0%`** Retention
  - **Mean Final Alignment Error**: **`6.2°`**
- **Range 2 ($40^\circ - 90^\circ$) Success Rate**: **`49.7%`** | Rotational Closure: **`60.6%`** | Grasp Retention: **`75.7%`**

---

### 🎓 Live Presentation Demo for Professor
To demonstrate the live in-hand manipulation in interactive 3D with the single green target arrow anchored to the object and 1.5-second firm hold lock-in:

```bash
python3 in_hand_manipulation/evaluate.py \
  --checkpoint best_model/best_model.pth \
  --render \
  --hold_time 1.5 \
  --max_goals 1 \
  --smooth_alpha 0.50 \
  --episodes_per_range 3 \
  --epsilon_ratio 0.10 \
  --sample_mode shuffle
```

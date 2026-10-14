# Best Champion Model (Grand Tournament Winner)

- **Checkpoint File**: `best_model.pth`
- **Promoted Model**: Stage 2F Iter 500 (Contender)
- **Source Checkpoint**: `ppo_close_contender_iter_500.pth`
- **Evaluation Protocol**: Comprehensive 6,000 Deterministic Episodes (1,000 per range per object)
- **Target Precision Envelope**: 10.0% Relative Error Tolerance (sub-9.0° precision)
- **Grand Tournament Score**: **0.7015**
- **Overall Success Rate**: **63.2%**
- **Overall Rotational Closure**: **71.1%**
- **Overall Grasp Retention**: **85.9%** (Drop Rate: **14.1%**)

### Stratified Angle Range Breakdown (3,000 episodes per range)
- **Range 1 (10° to 40° Small/Medium)**: 88.8% Success
- **Range 2 (40° to 90° Large Gaiting)**: 37.7% Success

### Object Breakdown (2,000 episodes per object)
- **Cylinder**: 58.2% Success
- **Cube**: 58.5% Success
- **Cuboid**: 73.1% Success

- **Last Verified**: 2026-10-14 20:40:04

======================================================================================
IN-HAND MANIPULATION SIM-TO-REAL DOMAIN RANDOMIZATION REPORT (@10.0% Tol)
Checkpoint: fine_tune_training/stage4_dr_full/ppo_final_champion_actor.pth
Total Episodes Evaluated: 600
DR Conditions:
  • Actuation Delay: 0 to 1 steps (0 to 20 ms lag)
  • Observation Sensor Noise Scale: 0.80
  • Fingertip/Palm Silicone Friction: μ in [2.1, 3.9]
  • Object Geometry / Mass / Friction: ±30%
  • Ground-Truth Physical Privileges: Disabled (Nominal Observation Only)
======================================================================================

######################################################################################
SECTION 1: RANGE 1 (10.0° to 40.0° Small-to-Medium Rotations under DR)
######################################################################################
Object       | Episodes | Success   | Closure   | Pre-Ret   | Pre-Drop  | Raw-Drop  | Final Err  | Steps 
---------------------------------------------------------------------------------------------------------
Cylinder     | 100      |   46.0%  |   46.0%  |   47.0%  |   53.0%  |   53.0%  |  68.8°     |  82.4
Cube         | 100      |   30.0%  |   40.8%  |   58.0%  |   42.0%  |   42.0%  |  55.4°     | 179.5
Cuboid       | 100      |   45.0%  |   48.9%  |   61.0%  |   39.0%  |   39.0%  |  46.0°     | 158.4
---------------------------------------------------------------------------------------------------------
RANGE 1 TOTAL | 300      |   40.3%  |   45.2%  |   55.3%  |   44.7%  |   44.7%  |  56.7°     | 140.1
  --> Range 1 Effective Manipulation Index (EMI): 0.2503 | Benchmark Score: 0.4480

######################################################################################
SECTION 2: RANGE 2 (40.0° to 90.0° Large-Angle In-Hand Gaiting under DR)
######################################################################################
Object       | Episodes | Success   | Closure   | Pre-Ret   | Pre-Drop  | Raw-Drop  | Final Err  | Steps 
---------------------------------------------------------------------------------------------------------
Cylinder     | 100      |   20.0%  |   22.0%  |   28.0%  |   72.0%  |   72.0%  |  98.7°     | 184.4
Cube         | 100      |   21.0%  |   28.9%  |   42.0%  |   58.0%  |   58.0%  |  76.1°     | 241.5
Cuboid       | 100      |   34.0%  |   38.0%  |   43.0%  |   57.0%  |   57.0%  |  69.1°     | 233.7
---------------------------------------------------------------------------------------------------------
RANGE 2 TOTAL | 300      |   25.0%  |   29.6%  |   37.7%  |   62.3%  |   62.3%  |  81.3°     | 219.9
  --> Range 2 Effective Manipulation Index (EMI): 0.1116 | Benchmark Score: 0.2893

======================================================================================
SECTION 3: GRAND COMPOSITE SIM-TO-REAL DR SUMMARY (FULL 10° TO 90° ENVELOPE)
======================================================================================
Evaluation Metric                   | Range 1 (10°-40°)    | Range 2 (40°-90°)    | Grand Total    
------------------------------------------------------------------------------------------------
Total Evaluated Episodes            | 300                  | 300                  | 600            
Success Rate (@10.0% Error)         | 40.3               % | 25.0               % | 32.7          %
Mean Rotational Closure             | 45.2               % | 29.6               % | 37.4          %
Pre-Goal Retention Rate             | 55.3               % | 37.7               % | 46.5          %
Pre-Goal Drop Rate                  | 44.7               % | 62.3               % | 53.5          %
Raw Drop Rate (All Steps)           | 44.7               % | 62.3               % | 53.5          %
Mean Final Geodesic Error           | 56.7               ° | 81.3               ° | 69.0          °
Mean Steps per Episode              | 140.1                | 219.9                | 180.0          
Peak Hardware Torque                | 21.64            N·m | 28.21            N·m | 28.21       N·m
Effective Manipulation Index (EMI)  | 0.2503               | 0.1116               | 0.1741         
Composite Benchmark Score           | 0.4480               | 0.2893               | 0.3686         
================================================================================================


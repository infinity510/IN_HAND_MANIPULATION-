======================================================================================
IN-HAND MANIPULATION SIM-TO-REAL DOMAIN RANDOMIZATION REPORT (@10.0% Tol)
Checkpoint: /mnt/Windows_SSD/Users/sheet/Desktop/AKSHAT/DC_PROJECT/fine_tune_training/stage14_breakthrough_70plus/ppo_final_champion_actor.pth
Total Episodes Evaluated: 600
DR Conditions:
  • Actuation Delay: 0 to 1 steps (0 to 20 ms lag)
  • ArUco Vision Latency: 1 to 2 steps (20 to 40 ms camera+pose lag)
  • Action Smoothing Filter Alpha: 0.50
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
Cylinder     | 100      |   69.0%  |   75.4%  |   87.0%  |   13.0%  |   13.0%  |  18.8°     | 150.0
Cube         | 100      |   66.0%  |   76.0%  |   92.0%  |    8.0%  |    8.0%  |  15.5°     | 189.8
Cuboid       | 100      |   83.0%  |   89.5%  |   96.0%  |    4.0%  |    4.0%  |   9.1°     | 179.1
---------------------------------------------------------------------------------------------------------
RANGE 1 TOTAL | 300      |   72.7%  |   80.3%  |   91.7%  |    8.3%  |    8.3%  |  14.4°     | 173.0
  --> Range 1 Effective Manipulation Index (EMI): 0.7360 | Benchmark Score: 0.7875

######################################################################################
SECTION 2: RANGE 2 (40.0° to 90.0° Large-Angle In-Hand Gaiting under DR)
######################################################################################
Object       | Episodes | Success   | Closure   | Pre-Ret   | Pre-Drop  | Raw-Drop  | Final Err  | Steps 
---------------------------------------------------------------------------------------------------------
Cylinder     | 100      |   54.0%  |   60.7%  |   73.0%  |   27.0%  |   27.0%  |  47.8°     | 299.8
Cube         | 100      |   28.0%  |   45.4%  |   78.0%  |   22.0%  |   22.0%  |  49.4°     | 456.2
Cuboid       | 100      |   52.0%  |   65.8%  |   76.0%  |   24.0%  |   24.0%  |  32.5°     | 317.0
---------------------------------------------------------------------------------------------------------
RANGE 2 TOTAL | 300      |   44.7%  |   57.3%  |   75.7%  |   24.3%  |   24.3%  |  43.2°     | 357.7
  --> Range 2 Effective Manipulation Index (EMI): 0.4334 | Benchmark Score: 0.5465

======================================================================================
SECTION 3: GRAND COMPOSITE SIM-TO-REAL DR SUMMARY (FULL 10° TO 90° ENVELOPE)
======================================================================================
Evaluation Metric                   | Range 1 (10°-40°)    | Range 2 (40°-90°)    | Grand Total    
------------------------------------------------------------------------------------------------
Total Evaluated Episodes            | 300                  | 300                  | 600            
Success Rate (@10.0% Error)         | 72.7               % | 44.7               % | 58.7          %
Mean Rotational Closure             | 80.3               % | 57.3               % | 68.8          %
Pre-Goal Retention Rate             | 91.7               % | 75.7               % | 83.7          %
Pre-Goal Drop Rate                  | 8.3                % | 24.3               % | 16.3          %
Raw Drop Rate (All Steps)           | 8.3                % | 24.3               % | 16.3          %
Mean Final Geodesic Error           | 14.4               ° | 43.2               ° | 28.8          °
Mean Steps per Episode              | 173.0                | 357.7                | 265.3          
Peak Hardware Torque                | 6.13             N·m | 5.68             N·m | 6.13        N·m
Effective Manipulation Index (EMI)  | 0.7360               | 0.4334               | 0.5755         
Composite Benchmark Score           | 0.7875               | 0.5465               | 0.6670         
================================================================================================


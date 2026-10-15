======================================================================================
IN-HAND MANIPULATION SIM-TO-REAL DOMAIN RANDOMIZATION REPORT (@10.0% Tol)
Checkpoint: best_model/best_model.pth
Total Episodes Evaluated: 150
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
Cylinder     | 25       |   88.0%  |   90.9%  |   96.0%  |    4.0%  |    4.0%  |   9.0°     | 107.0
Cube         | 25       |   80.0%  |   88.3%  |   96.0%  |    4.0%  |    4.0%  |   8.7°     | 169.6
Cuboid       | 25       |   88.0%  |   93.4%  |  100.0%  |    0.0%  |    0.0%  |   3.3°     | 178.2
---------------------------------------------------------------------------------------------------------
RANGE 1 TOTAL | 75       |   85.3%  |   90.8%  |   97.3%  |    2.7%  |    2.7%  |   7.0°     | 151.6
  --> Range 1 Effective Manipulation Index (EMI): 0.8842 | Benchmark Score: 0.8939

######################################################################################
SECTION 2: RANGE 2 (40.0° to 90.0° Large-Angle In-Hand Gaiting under DR)
######################################################################################
Object       | Episodes | Success   | Closure   | Pre-Ret   | Pre-Drop  | Raw-Drop  | Final Err  | Steps 
---------------------------------------------------------------------------------------------------------
Cylinder     | 25       |   60.0%  |   62.8%  |   68.0%  |   32.0%  |   32.0%  |  48.9°     | 246.4
Cube         | 25       |   28.0%  |   53.7%  |   88.0%  |   12.0%  |   12.0%  |  40.5°     | 480.0
Cuboid       | 25       |   36.0%  |   56.3%  |   72.0%  |   28.0%  |   28.0%  |  34.3°     | 429.6
---------------------------------------------------------------------------------------------------------
RANGE 2 TOTAL | 75       |   41.3%  |   57.6%  |   76.0%  |   24.0%  |   24.0%  |  41.2°     | 385.4
  --> Range 2 Effective Manipulation Index (EMI): 0.4378 | Benchmark Score: 0.5315

======================================================================================
SECTION 3: GRAND COMPOSITE SIM-TO-REAL DR SUMMARY (FULL 10° TO 90° ENVELOPE)
======================================================================================
Evaluation Metric                   | Range 1 (10°-40°)    | Range 2 (40°-90°)    | Grand Total    
------------------------------------------------------------------------------------------------
Total Evaluated Episodes            | 75                   | 75                   | 150            
Success Rate (@10.0% Error)         | 85.3               % | 41.3               % | 63.3          %
Mean Rotational Closure             | 90.8               % | 57.6               % | 74.2          %
Pre-Goal Retention Rate             | 97.3               % | 76.0               % | 86.7          %
Pre-Goal Drop Rate                  | 2.7                % | 24.0               % | 13.3          %
Raw Drop Rate (All Steps)           | 2.7                % | 24.0               % | 13.3          %
Mean Final Geodesic Error           | 7.0                ° | 41.2               ° | 24.1          °
Mean Steps per Episode              | 151.6                | 385.4                | 268.5          
Peak Hardware Torque                | 5.26             N·m | 4.79             N·m | 5.26        N·m
Effective Manipulation Index (EMI)  | 0.8842               | 0.4378               | 0.6432         
Composite Benchmark Score           | 0.8939               | 0.5315               | 0.7127         
================================================================================================


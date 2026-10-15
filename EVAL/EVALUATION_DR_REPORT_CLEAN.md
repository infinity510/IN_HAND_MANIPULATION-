======================================================================================
IN-HAND MANIPULATION SIM-TO-REAL DOMAIN RANDOMIZATION REPORT (@10.0% Tol)
Checkpoint: best_model/best_model_clean.pth
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
Cylinder     | 25       |   72.0%  |   79.1%  |   84.0%  |   16.0%  |   16.0%  |  16.7°     | 154.6
Cube         | 25       |   76.0%  |   82.6%  |   92.0%  |    8.0%  |    8.0%  |   9.6°     | 202.2
Cuboid       | 25       |   56.0%  |   68.8%  |   96.0%  |    4.0%  |    4.0%  |  11.1°     | 237.1
---------------------------------------------------------------------------------------------------------
RANGE 1 TOTAL | 75       |   68.0%  |   76.8%  |   90.7%  |    9.3%  |    9.3%  |  12.5°     | 197.9
  --> Range 1 Effective Manipulation Index (EMI): 0.6966 | Benchmark Score: 0.7518

######################################################################################
SECTION 2: RANGE 2 (40.0° to 90.0° Large-Angle In-Hand Gaiting under DR)
######################################################################################
Object       | Episodes | Success   | Closure   | Pre-Ret   | Pre-Drop  | Raw-Drop  | Final Err  | Steps 
---------------------------------------------------------------------------------------------------------
Cylinder     | 25       |   32.0%  |   43.1%  |   60.0%  |   40.0%  |   40.0%  |  61.4°     | 342.5
Cube         | 25       |   24.0%  |   46.3%  |   72.0%  |   28.0%  |   28.0%  |  51.0°     | 491.1
Cuboid       | 25       |   48.0%  |   54.1%  |   68.0%  |   32.0%  |   32.0%  |  43.8°     | 377.8
---------------------------------------------------------------------------------------------------------
RANGE 2 TOTAL | 75       |   34.7%  |   47.8%  |   66.7%  |   33.3%  |   33.3%  |  52.1°     | 403.8
  --> Range 2 Effective Manipulation Index (EMI): 0.3188 | Benchmark Score: 0.4501

======================================================================================
SECTION 3: GRAND COMPOSITE SIM-TO-REAL DR SUMMARY (FULL 10° TO 90° ENVELOPE)
======================================================================================
Evaluation Metric                   | Range 1 (10°-40°)    | Range 2 (40°-90°)    | Grand Total    
------------------------------------------------------------------------------------------------
Total Evaluated Episodes            | 75                   | 75                   | 150            
Success Rate (@10.0% Error)         | 68.0               % | 34.7               % | 51.3          %
Mean Rotational Closure             | 76.8               % | 47.8               % | 62.3          %
Pre-Goal Retention Rate             | 90.7               % | 66.7               % | 78.7          %
Pre-Goal Drop Rate                  | 9.3                % | 33.3               % | 21.3          %
Raw Drop Rate (All Steps)           | 9.3                % | 33.3               % | 21.3          %
Mean Final Geodesic Error           | 12.5               ° | 52.1               ° | 32.3          °
Mean Steps per Episode              | 197.9                | 403.8                | 300.9          
Peak Hardware Torque                | 5.73             N·m | 5.03             N·m | 5.73        N·m
Effective Manipulation Index (EMI)  | 0.6966               | 0.3188               | 0.4903         
Composite Benchmark Score           | 0.7518               | 0.4501               | 0.6010         
================================================================================================


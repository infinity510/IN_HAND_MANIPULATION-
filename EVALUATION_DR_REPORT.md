======================================================================================
IN-HAND MANIPULATION SIM-TO-REAL DOMAIN RANDOMIZATION REPORT (@10.0% Tol)
Checkpoint: best_model/best_model.pth
Total Episodes Evaluated: 120
DR Conditions:
  • Actuation Delay: 0 to 1 steps (0 to 20 ms lag)
  • ArUco Vision Latency: 1 to 2 steps (20 to 40 ms camera+pose lag)
  • Action Smoothing Filter Alpha: 0.70
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
Cylinder     | 20       |   90.0%  |   90.0%  |   90.0%  |   10.0%  |   10.0%  |  13.5°     |  99.5
Cube         | 20       |   70.0%  |   80.4%  |  100.0%  |    0.0%  |    0.0%  |   5.2°     | 224.2
Cuboid       | 20       |   80.0%  |   87.0%  |  100.0%  |    0.0%  |    0.0%  |   5.9°     | 183.2
---------------------------------------------------------------------------------------------------------
RANGE 1 TOTAL | 60       |   80.0%  |   85.8%  |   96.7%  |    3.3%  |    3.3%  |   8.2°     | 169.0
  --> Range 1 Effective Manipulation Index (EMI): 0.8295 | Benchmark Score: 0.8508

######################################################################################
SECTION 2: RANGE 2 (40.0° to 90.0° Large-Angle In-Hand Gaiting under DR)
######################################################################################
Object       | Episodes | Success   | Closure   | Pre-Ret   | Pre-Drop  | Raw-Drop  | Final Err  | Steps 
---------------------------------------------------------------------------------------------------------
Cylinder     | 20       |   55.0%  |   70.2%  |   80.0%  |   20.0%  |   20.0%  |  37.9°     | 315.1
Cube         | 20       |   35.0%  |   55.5%  |   80.0%  |   20.0%  |   20.0%  |  38.3°     | 428.6
Cuboid       | 20       |   50.0%  |   57.6%  |   65.0%  |   35.0%  |   35.0%  |  49.1°     | 350.4
---------------------------------------------------------------------------------------------------------
RANGE 2 TOTAL | 60       |   46.7%  |   61.1%  |   75.0%  |   25.0%  |   25.0%  |  41.8°     | 364.7
  --> Range 2 Effective Manipulation Index (EMI): 0.4582 | Benchmark Score: 0.5666

======================================================================================
SECTION 3: GRAND COMPOSITE SIM-TO-REAL DR SUMMARY (FULL 10° TO 90° ENVELOPE)
======================================================================================
Evaluation Metric                   | Range 1 (10°-40°)    | Range 2 (40°-90°)    | Grand Total    
------------------------------------------------------------------------------------------------
Total Evaluated Episodes            | 60                   | 60                   | 120            
Success Rate (@10.0% Error)         | 80.0               % | 46.7               % | 63.3          %
Mean Rotational Closure             | 85.8               % | 61.1               % | 73.4          %
Pre-Goal Retention Rate             | 96.7               % | 75.0               % | 85.8          %
Pre-Goal Drop Rate                  | 3.3                % | 25.0               % | 14.2          %
Raw Drop Rate (All Steps)           | 3.3                % | 25.0               % | 14.2          %
Mean Final Geodesic Error           | 8.2                ° | 41.8               ° | 25.0          °
Mean Steps per Episode              | 169.0                | 364.7                | 266.8          
Peak Hardware Torque                | 5.58             N·m | 5.01             N·m | 5.58        N·m
Effective Manipulation Index (EMI)  | 0.8295               | 0.4582               | 0.6304         
Composite Benchmark Score           | 0.8508               | 0.5666               | 0.7087         
================================================================================================


======================================================================================
IN-HAND MANIPULATION SIM-TO-REAL DOMAIN RANDOMIZATION REPORT (@10.0% Tol)
Checkpoint: best_model/best_model_dr_peak_07246.pth
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
Cylinder     | 25       |   92.0%  |   92.0%  |   92.0%  |    8.0%  |    8.0%  |   9.6°     |  83.8
Cube         | 25       |   80.0%  |   85.2%  |  100.0%  |    0.0%  |    0.0%  |   6.8°     | 186.7
Cuboid       | 25       |   72.0%  |   84.8%  |   96.0%  |    4.0%  |    4.0%  |   7.3°     | 209.8
---------------------------------------------------------------------------------------------------------
RANGE 1 TOTAL | 75       |   81.3%  |   87.3%  |   96.0%  |    4.0%  |    4.0%  |   7.9°     | 160.1
  --> Range 1 Effective Manipulation Index (EMI): 0.8384 | Benchmark Score: 0.8607

######################################################################################
SECTION 2: RANGE 2 (40.0° to 90.0° Large-Angle In-Hand Gaiting under DR)
######################################################################################
Object       | Episodes | Success   | Closure   | Pre-Ret   | Pre-Drop  | Raw-Drop  | Final Err  | Steps 
---------------------------------------------------------------------------------------------------------
Cylinder     | 25       |   52.0%  |   60.3%  |   64.0%  |   36.0%  |   36.0%  |  51.1°     | 265.4
Cube         | 25       |   28.0%  |   52.8%  |   88.0%  |   12.0%  |   12.0%  |  39.2°     | 487.7
Cuboid       | 25       |   40.0%  |   55.8%  |   64.0%  |   36.0%  |   36.0%  |  49.0°     | 393.2
---------------------------------------------------------------------------------------------------------
RANGE 2 TOTAL | 75       |   40.0%  |   56.3%  |   72.0%  |   28.0%  |   28.0%  |  46.4°     | 382.1
  --> Range 2 Effective Manipulation Index (EMI): 0.4052 | Benchmark Score: 0.5128

======================================================================================
SECTION 3: GRAND COMPOSITE SIM-TO-REAL DR SUMMARY (FULL 10° TO 90° ENVELOPE)
======================================================================================
Evaluation Metric                   | Range 1 (10°-40°)    | Range 2 (40°-90°)    | Grand Total    
------------------------------------------------------------------------------------------------
Total Evaluated Episodes            | 75                   | 75                   | 150            
Success Rate (@10.0% Error)         | 81.3               % | 40.0               % | 60.7          %
Mean Rotational Closure             | 87.3               % | 56.3               % | 71.8          %
Pre-Goal Retention Rate             | 96.0               % | 72.0               % | 84.0          %
Pre-Goal Drop Rate                  | 4.0                % | 28.0               % | 16.0          %
Raw Drop Rate (All Steps)           | 4.0                % | 28.0               % | 16.0          %
Mean Final Geodesic Error           | 7.9                ° | 46.4               ° | 27.2          °
Mean Steps per Episode              | 160.1                | 382.1                | 271.1          
Peak Hardware Torque                | 5.11             N·m | 4.92             N·m | 5.11        N·m
Effective Manipulation Index (EMI)  | 0.8384               | 0.4052               | 0.6031         
Composite Benchmark Score           | 0.8607               | 0.5128               | 0.6867         
================================================================================================


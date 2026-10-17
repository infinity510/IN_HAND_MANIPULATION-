======================================================================================
IN-HAND MANIPULATION SIM-TO-REAL DOMAIN RANDOMIZATION REPORT (@15.0% Tol)
Checkpoint: /mnt/Windows_SSD/Users/sheet/Desktop/AKSHAT/DC_PROJECT/best_model/best_model.pth
Total Episodes Evaluated: 12
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
Cylinder     | 3        |  100.0%  |  100.0%  |  100.0%  |    0.0%  |    0.0%  |   3.5°     |  70.0
Cube         | 3        |  100.0%  |  100.0%  |  100.0%  |    0.0%  |    0.0%  |   1.9°     |  80.0
Cuboid       | 3        |  100.0%  |  100.0%  |  100.0%  |    0.0%  |    0.0%  |   2.3°     | 164.3
---------------------------------------------------------------------------------------------------------
RANGE 1 TOTAL | 9        |  100.0%  |  100.0%  |  100.0%  |    0.0%  |    0.0%  |   2.6°     | 104.8
  --> Range 1 Effective Manipulation Index (EMI): 1.0000 | Benchmark Score: 1.0000

######################################################################################
SECTION 2: RANGE 2 (40.0° to 90.0° Large-Angle In-Hand Gaiting under DR)
######################################################################################
Object       | Episodes | Success   | Closure   | Pre-Ret   | Pre-Drop  | Raw-Drop  | Final Err  | Steps 
---------------------------------------------------------------------------------------------------------
Cylinder     | 2        |   50.0%  |   64.8%  |  100.0%  |    0.0%  |    0.0%  |  28.4°     | 537.5
Cuboid       | 1        |  100.0%  |  100.0%  |  100.0%  |    0.0%  |    0.0%  |   7.4°     | 212.0
---------------------------------------------------------------------------------------------------------
RANGE 2 TOTAL | 3        |   66.7%  |   76.5%  |  100.0%  |    0.0%  |    0.0%  |  21.4°     | 429.0
  --> Range 2 Effective Manipulation Index (EMI): 0.7653 | Benchmark Score: 0.7629

======================================================================================
SECTION 3: GRAND COMPOSITE SIM-TO-REAL DR SUMMARY (FULL 10° TO 90° ENVELOPE)
======================================================================================
Evaluation Metric                   | Range 1 (10°-40°)    | Range 2 (40°-90°)    | Grand Total    
------------------------------------------------------------------------------------------------
Total Evaluated Episodes            | 9                    | 3                    | 12             
Success Rate (@15.0% Error)         | 100.0              % | 66.7               % | 91.7          %
Mean Rotational Closure             | 100.0              % | 76.5               % | 94.1          %
Pre-Goal Retention Rate             | 100.0              % | 100.0              % | 100.0         %
Pre-Goal Drop Rate                  | 0.0                % | 0.0                % | 0.0           %
Raw Drop Rate (All Steps)           | 0.0                % | 0.0                % | 0.0           %
Mean Final Geodesic Error           | 2.6                ° | 21.4               ° | 7.3           °
Mean Steps per Episode              | 104.8                | 429.0                | 185.8          
Peak Hardware Torque                | 4.10             N·m | 4.80             N·m | 4.80        N·m
Effective Manipulation Index (EMI)  | 1.0000               | 0.7653               | 0.9413         
Composite Benchmark Score           | 1.0000               | 0.7629               | 0.9407         
================================================================================================


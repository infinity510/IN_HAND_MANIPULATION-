======================================================================================
IN-HAND MANIPULATION SIM-TO-REAL DOMAIN RANDOMIZATION REPORT (@10.0% Tol)
Checkpoint: /mnt/Windows_SSD/Users/sheet/Desktop/AKSHAT/DC_PROJECT/fine_tune_training/stage5_dr_vision_delay/ppo_eval_best_actor.pth
Total Episodes Evaluated: 600
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
Cylinder     | 100      |   71.0%  |   78.0%  |   85.0%  |   15.0%  |   15.0%  |  16.4°     | 145.3
Cube         | 100      |   61.0%  |   76.8%  |   94.0%  |    6.0%  |    6.0%  |  13.3°     | 220.1
Cuboid       | 100      |   62.0%  |   77.6%  |   97.0%  |    3.0%  |    3.0%  |   9.4°     | 237.6
---------------------------------------------------------------------------------------------------------
RANGE 1 TOTAL | 300      |   64.7%  |   77.5%  |   92.0%  |    8.0%  |    8.0%  |  13.0°     | 201.0
  --> Range 1 Effective Manipulation Index (EMI): 0.7128 | Benchmark Score: 0.7398

######################################################################################
SECTION 2: RANGE 2 (40.0° to 90.0° Large-Angle In-Hand Gaiting under DR)
######################################################################################
Object       | Episodes | Success   | Closure   | Pre-Ret   | Pre-Drop  | Raw-Drop  | Final Err  | Steps 
---------------------------------------------------------------------------------------------------------
Cylinder     | 100      |   40.0%  |   53.9%  |   72.0%  |   28.0%  |   28.0%  |  54.7°     | 367.3
Cube         | 100      |   21.0%  |   42.6%  |   75.0%  |   25.0%  |   25.0%  |  48.4°     | 491.4
Cuboid       | 100      |   44.0%  |   59.8%  |   70.0%  |   30.0%  |   30.0%  |  34.6°     | 343.9
---------------------------------------------------------------------------------------------------------
RANGE 2 TOTAL | 300      |   35.0%  |   52.1%  |   72.3%  |   27.7%  |   27.7%  |  45.9°     | 400.9
  --> Range 2 Effective Manipulation Index (EMI): 0.3767 | Benchmark Score: 0.4759

======================================================================================
SECTION 3: GRAND COMPOSITE SIM-TO-REAL DR SUMMARY (FULL 10° TO 90° ENVELOPE)
======================================================================================
Evaluation Metric                   | Range 1 (10°-40°)    | Range 2 (40°-90°)    | Grand Total    
------------------------------------------------------------------------------------------------
Total Evaluated Episodes            | 300                  | 300                  | 600            
Success Rate (@10.0% Error)         | 64.7               % | 35.0               % | 49.8          %
Mean Rotational Closure             | 77.5               % | 52.1               % | 64.8          %
Pre-Goal Retention Rate             | 92.0               % | 72.3               % | 82.2          %
Pre-Goal Drop Rate                  | 8.0                % | 27.7               % | 17.8          %
Raw Drop Rate (All Steps)           | 8.0                % | 27.7               % | 17.8          %
Mean Final Geodesic Error           | 13.0               ° | 45.9               ° | 29.4          °
Mean Steps per Episode              | 201.0                | 400.9                | 301.0          
Peak Hardware Torque                | 6.11             N·m | 6.26             N·m | 6.26        N·m
Effective Manipulation Index (EMI)  | 0.7128               | 0.3767               | 0.5323         
Composite Benchmark Score           | 0.7398               | 0.4759               | 0.6078         
================================================================================================


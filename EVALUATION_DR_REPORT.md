======================================================================================
IN-HAND MANIPULATION SIM-TO-REAL DOMAIN RANDOMIZATION REPORT (@10.0% Tol)
Checkpoint: best_model/best_model.pth
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
Cylinder     | 100      |   40.0%  |   41.5%  |   45.0%  |   55.0%  |   55.0%  |  71.5°     |  87.8
Cube         | 100      |   33.0%  |   39.4%  |   56.0%  |   44.0%  |   44.0%  |  61.6°     | 163.2
Cuboid       | 100      |   47.0%  |   49.6%  |   58.0%  |   42.0%  |   42.0%  |  46.8°     | 134.2
---------------------------------------------------------------------------------------------------------
RANGE 1 TOTAL | 300      |   40.0%  |   43.5%  |   53.0%  |   47.0%  |   47.0%  |  60.0°     | 128.4
  --> Range 1 Effective Manipulation Index (EMI): 0.2306 | Benchmark Score: 0.4365

######################################################################################
SECTION 2: RANGE 2 (40.0° to 90.0° Large-Angle In-Hand Gaiting under DR)
######################################################################################
Object       | Episodes | Success   | Closure   | Pre-Ret   | Pre-Drop  | Raw-Drop  | Final Err  | Steps 
---------------------------------------------------------------------------------------------------------
Cylinder     | 100      |   29.0%  |   31.6%  |   35.0%  |   65.0%  |   65.0%  |  84.4°     | 160.4
Cube         | 100      |   19.0%  |   24.3%  |   34.0%  |   66.0%  |   66.0%  |  77.1°     | 203.7
Cuboid       | 100      |   32.0%  |   34.9%  |   39.0%  |   61.0%  |   61.0%  |  67.6°     | 189.2
---------------------------------------------------------------------------------------------------------
RANGE 2 TOTAL | 300      |   26.7%  |   30.3%  |   36.0%  |   64.0%  |   64.0%  |  76.3°     | 184.4
  --> Range 2 Effective Manipulation Index (EMI): 0.1089 | Benchmark Score: 0.2961

======================================================================================
SECTION 3: GRAND COMPOSITE SIM-TO-REAL DR SUMMARY (FULL 10° TO 90° ENVELOPE)
======================================================================================
Evaluation Metric                   | Range 1 (10°-40°)    | Range 2 (40°-90°)    | Grand Total    
------------------------------------------------------------------------------------------------
Total Evaluated Episodes            | 300                  | 300                  | 600            
Success Rate (@10.0% Error)         | 40.0               % | 26.7               % | 33.3          %
Mean Rotational Closure             | 43.5               % | 30.3               % | 36.9          %
Pre-Goal Retention Rate             | 53.0               % | 36.0               % | 44.5          %
Pre-Goal Drop Rate                  | 47.0               % | 64.0               % | 55.5          %
Raw Drop Rate (All Steps)           | 47.0               % | 64.0               % | 55.5          %
Mean Final Geodesic Error           | 60.0               ° | 76.3               ° | 68.2          °
Mean Steps per Episode              | 128.4                | 184.4                | 156.4          
Peak Hardware Torque                | 21.60            N·m | 34.12            N·m | 34.12       N·m
Effective Manipulation Index (EMI)  | 0.2306               | 0.1089               | 0.1641         
Composite Benchmark Score           | 0.4365               | 0.2961               | 0.3663         
================================================================================================


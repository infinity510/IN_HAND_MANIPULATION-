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
Cylinder     | 100      |   38.0%  |   40.1%  |   44.0%  |   56.0%  |   56.0%  |  72.5°     |  90.9
Cube         | 100      |   32.0%  |   40.2%  |   55.0%  |   45.0%  |   45.0%  |  63.0°     | 164.4
Cuboid       | 100      |   46.0%  |   49.0%  |   58.0%  |   42.0%  |   42.0%  |  47.6°     | 135.9
---------------------------------------------------------------------------------------------------------
RANGE 1 TOTAL | 300      |   38.7%  |   43.1%  |   52.3%  |   47.7%  |   47.7%  |  61.0°     | 130.4
  --> Range 1 Effective Manipulation Index (EMI): 0.2255 | Benchmark Score: 0.4273

######################################################################################
SECTION 2: RANGE 2 (40.0° to 90.0° Large-Angle In-Hand Gaiting under DR)
######################################################################################
Object       | Episodes | Success   | Closure   | Pre-Ret   | Pre-Drop  | Raw-Drop  | Final Err  | Steps 
---------------------------------------------------------------------------------------------------------
Cylinder     | 100      |   29.0%  |   31.9%  |   35.0%  |   65.0%  |   65.0%  |  84.5°     | 158.3
Cube         | 100      |   19.0%  |   24.0%  |   32.0%  |   68.0%  |   68.0%  |  78.2°     | 196.9
Cuboid       | 100      |   33.0%  |   35.7%  |   39.0%  |   61.0%  |   61.0%  |  68.0°     | 187.6
---------------------------------------------------------------------------------------------------------
RANGE 2 TOTAL | 300      |   27.0%  |   30.5%  |   35.3%  |   64.7%  |   64.7%  |  76.9°     | 181.0
  --> Range 2 Effective Manipulation Index (EMI): 0.1079 | Benchmark Score: 0.2972

======================================================================================
SECTION 3: GRAND COMPOSITE SIM-TO-REAL DR SUMMARY (FULL 10° TO 90° ENVELOPE)
======================================================================================
Evaluation Metric                   | Range 1 (10°-40°)    | Range 2 (40°-90°)    | Grand Total    
------------------------------------------------------------------------------------------------
Total Evaluated Episodes            | 300                  | 300                  | 600            
Success Rate (@10.0% Error)         | 38.7               % | 27.0               % | 32.8          %
Mean Rotational Closure             | 43.1               % | 30.5               % | 36.8          %
Pre-Goal Retention Rate             | 52.3               % | 35.3               % | 43.8          %
Pre-Goal Drop Rate                  | 47.7               % | 64.7               % | 56.2          %
Raw Drop Rate (All Steps)           | 47.7               % | 64.7               % | 56.2          %
Mean Final Geodesic Error           | 61.0               ° | 76.9               ° | 69.0          °
Mean Steps per Episode              | 130.4                | 181.0                | 155.7          
Peak Hardware Torque                | 21.60            N·m | 34.12            N·m | 34.12       N·m
Effective Manipulation Index (EMI)  | 0.2255               | 0.1079               | 0.1613         
Composite Benchmark Score           | 0.4273               | 0.2972               | 0.3623         
================================================================================================


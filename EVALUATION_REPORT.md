======================================================================================
IN-HAND MANIPULATION COMPREHENSIVE STRATIFIED BENCHMARK REPORT
Model Checkpoint: best_model/best_model.pth (MD5: 7db0136c99d99575807b93e9c3bd0d4a)
Gripper: 12-DOF Delto 3-Finger Hand | Physics: MuJoCo
======================================================================================

PART I: STANDARD BENCHMARK (@10.0% Relative Error Tolerance / Sub-4.3° Precision)
Total Evaluated Episodes: 600 (100 episodes per range per object)
--------------------------------------------------------------------------------------

SECTION 1: RANGE 1 (10.0° to 40.0° Small-to-Medium Rotations)
Object       | Episodes | Success   | Closure   | Pre-Ret   | Pre-Drop  | Raw-Drop  | Final Err  | Steps 
---------------------------------------------------------------------------------------------------------
Cylinder     | 100      |   90.0%  |   95.8%  |   99.0%  |    1.0%  |    1.0%  |   3.7°     | 115.8
Cube         | 100      |   77.0%  |   89.5%  |   99.0%  |    1.0%  |    1.0%  |   5.5°     | 187.1
Cuboid       | 100      |   75.0%  |   89.9%  |   99.0%  |    1.0%  |    1.0%  |   4.8°     | 204.2
---------------------------------------------------------------------------------------------------------
RANGE 1 TOTAL | 300      |   80.7%  |   91.8%  |   99.0%  |    1.0%  |    1.0%  |   4.7°     | 169.1
  --> Range 1 Effective Manipulation Index (EMI): 0.9084 | Benchmark Score: 0.8766

SECTION 2: RANGE 2 (40.0° to 90.0° Large-Angle In-Hand Gaiting)
Object       | Episodes | Success   | Closure   | Pre-Ret   | Pre-Drop  | Raw-Drop  | Final Err  | Steps 
---------------------------------------------------------------------------------------------------------
Cylinder     | 100      |   52.0%  |   67.9%  |   82.0%  |   18.0%  |   18.0%  |  39.7°     | 380.3
Cube         | 100      |   41.0%  |   55.7%  |   85.0%  |   15.0%  |   15.0%  |  35.1°     | 451.8
Cuboid       | 100      |   57.0%  |   69.7%  |   81.0%  |   19.0%  |   19.0%  |  23.5°     | 324.5
---------------------------------------------------------------------------------------------------------
RANGE 2 TOTAL | 300      |   50.0%  |   64.4%  |   82.7%  |   17.3%  |   17.3%  |  32.8°     | 385.5
  --> Range 2 Effective Manipulation Index (EMI): 0.5327 | Benchmark Score: 0.6086

GRAND COMPOSITE SUMMARY (@10.0% TOLERANCE)
Evaluation Metric                   | Range 1 (10°-40°)    | Range 2 (40°-90°)    | Grand Total    
------------------------------------------------------------------------------------------------
Total Evaluated Episodes            | 300                  | 300                  | 600            
Success Rate (@10% Error)           | 80.7%                | 50.0%                | 65.3%          
Mean Rotational Closure             | 91.8%                | 64.4%                | 78.1%          
Pre-Goal Retention Rate             | 99.0%                | 82.7%                | 90.8%          
Pre-Goal Drop Rate                  | 1.0%                 | 17.3%                | 9.2%           
Raw Drop Rate (All Steps)           | 1.0%                 | 17.3%                | 9.2%           
Mean Final Geodesic Error           | 4.7°                 | 32.8°                | 18.7°          
Mean Steps per Episode              | 169.1                | 385.5                | 277.3          
Peak Hardware Torque                | 4.93 N·m             | 5.63 N·m             | 5.63 N·m       
Effective Manipulation Index (EMI)  | 0.9084               | 0.5327               | 0.7094         
Composite Benchmark Score           | 0.8766               | 0.6086               | 0.7426         

======================================================================================
PART II: ULTRA-STRICT BENCHMARK (@5.0% Relative Error Tolerance / Sub-2.5° Precision)
Total Evaluated Episodes: 3600 (600 episodes per range per object)
--------------------------------------------------------------------------------------

SECTION 1: RANGE 1 (10.0° to 40.0° Small-to-Medium Rotations)
Object       | Episodes | Success   | Closure   | Pre-Ret   | Pre-Drop  | Raw-Drop  | Final Err  | Steps 
---------------------------------------------------------------------------------------------------------
Cylinder     | 600      |   53.2%  |   76.6%  |   95.0%  |    5.0%  |    5.0%  |   9.6°     | 236.1
Cube         | 600      |   43.0%  |   75.3%  |   97.7%  |    2.3%  |    2.3%  |   8.6°     | 280.5
Cuboid       | 600      |   51.7%  |   82.6%  |   98.0%  |    2.0%  |    2.0%  |   6.6°     | 263.7
---------------------------------------------------------------------------------------------------------
RANGE 1 TOTAL | 1800     |   49.3%  |   78.2%  |   96.9%  |    3.1%  |    3.1%  |   8.3°     | 260.1
  --> Range 1 Effective Manipulation Index (EMI): 0.7576 | Benchmark Score: 0.6747

SECTION 2: RANGE 2 (40.0° to 90.0° Large-Angle In-Hand Gaiting)
Object       | Episodes | Success   | Closure   | Pre-Ret   | Pre-Drop  | Raw-Drop  | Final Err  | Steps 
---------------------------------------------------------------------------------------------------------
Cylinder     | 600      |   33.8%  |   54.1%  |   68.3%  |   31.7%  |   31.7%  |  52.8°     | 431.3
Cube         | 600      |   24.3%  |   44.0%  |   72.8%  |   27.2%  |   27.2%  |  44.1°     | 480.9
Cuboid       | 600      |   33.7%  |   49.2%  |   62.3%  |   37.7%  |   37.7%  |  40.2°     | 384.0
---------------------------------------------------------------------------------------------------------
RANGE 2 TOTAL | 1800     |   30.6%  |   49.1%  |   67.8%  |   32.2%  |   32.2%  |  45.7°     | 432.1
  --> Range 2 Effective Manipulation Index (EMI): 0.3330 | Benchmark Score: 0.4360

GRAND COMPOSITE SUMMARY (@5.0% TOLERANCE)
Evaluation Metric                   | Range 1 (10°-40°)    | Range 2 (40°-90°)    | Grand Total    
------------------------------------------------------------------------------------------------
Total Evaluated Episodes            | 1800                 | 1800                 | 3600           
Success Rate (@5% Error)            | 49.3%                | 30.6%                | 39.9%          
Mean Rotational Closure             | 78.2%                | 49.1%                | 63.6%          
Pre-Goal Retention Rate             | 96.9%                | 67.8%                | 82.4%          
Pre-Goal Drop Rate                  | 3.1%                 | 32.2%                | 17.6%          
Raw Drop Rate (All Steps)           | 3.1%                 | 32.2%                | 17.6%          
Mean Final Geodesic Error           | 8.3°                 | 45.7°                | 27.0°          
Mean Steps per Episode              | 260.1                | 432.1                | 346.1          
Peak Hardware Torque                | 5.51 N·m             | 5.65 N·m             | 5.65 N·m       
Effective Manipulation Index (EMI)  | 0.7576               | 0.3330               | 0.5241         
Composite Benchmark Score           | 0.6747               | 0.4360               | 0.5554         
======================================================================================

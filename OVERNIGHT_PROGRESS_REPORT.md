# Overnight Master Curriculum Execution Report


## Stage 1A (EpsR = 25% Baseline) (2026-10-12 03:19:01)

```
nal Error: 34.24° | Rotational Closure:  14.8% | Status: SUCCESS (<= 4.0°)

[Episode 22/25] Object: 1 | Initial Target Angle:  23.2° | Success Threshold (10%): <= 2.32° | Horizon: 285 steps (5.70s)
  Finished: Steps: 250/285 | Final Error: 68.60° | Rotational Closure:   0.0% | Status: DROPPED

[Episode 23/25] Object: 1 | Initial Target Angle:  34.0° | Success Threshold (10%): <= 3.40° | Horizon: 341 steps (6.82s)
  Finished: Steps: 151/341 | Final Error: 105.21° | Rotational Closure:   0.0% | Status: DROPPED

[Episode 24/25] Object: 3 | Initial Target Angle:  24.4° | Success Threshold (10%): <= 2.44° | Horizon: 291 steps (5.82s)
  Finished: Steps: 291/291 | Final Error: 20.57° | Rotational Closure:  15.8% | Status: TIMEOUT (20.6° > 2.4°)

[Episode 25/25] Object: 3 | Initial Target Angle:  32.7° | Success Threshold (10%): <= 3.27° | Horizon: 334 steps (6.68s)
  Finished: Steps: 500/334 | Final Error:  8.96° | Rotational Closure:  72.6% | Status: SUCCESS (<= 3.3°)

======================================================================
STAGE 1 BENCHMARK SUMMARY (25 Episodes Completed):
Effective Manipulation Index (EMI): 0.1715  (Retention: 80.0% x Closure: 21.4%)
Success Rate (<= 10% error):    28.0% (7/25)
Mean Rotational Closure:             21.4%
Drop Rate:                           20.0% (5/25)
Mean Episode Steps:                 334.5 steps (6.69s)
Mean Angular Error:                 35.38° +/- 33.60°
======================================================================


```


## Stage 1B (2026-10-12 06:30:42)

```
sure:  21.2% | Status: TIMEOUT (31.7° > 4.0°)

[Episode 22/25] Object: 3 | Initial Target Angle:  24.8° | Success Threshold (10%): <= 2.48° | Horizon: 293 steps (5.86s)
  Finished: Steps: 293/293 | Final Error: 13.70° | Rotational Closure:  44.8% | Status: TIMEOUT (13.7° > 2.5°)

[Episode 23/25] Object: 1 | Initial Target Angle:  27.2° | Success Threshold (10%): <= 2.72° | Horizon: 304 steps (6.08s)
  Finished: Steps: 304/304 | Final Error: 10.10° | Rotational Closure:  62.8% | Status: TIMEOUT (10.1° > 2.7°)

[Episode 24/25] Object: 3 | Initial Target Angle:  16.2° | Success Threshold (10%): <= 1.62° | Horizon: 254 steps (5.08s)
  Finished: Steps: 404/254 | Final Error: 18.82° | Rotational Closure:   0.0% | Status: SUCCESS (<= 1.6°)

[Episode 25/25] Object: 1 | Initial Target Angle:  40.9° | Success Threshold (10%): <= 4.09° | Horizon: 383 steps (7.66s)
  Finished: Steps: 383/383 | Final Error: 23.22° | Rotational Closure:  43.3% | Status: TIMEOUT (23.2° > 4.1°)

======================================================================
STAGE 1 BENCHMARK SUMMARY (25 Episodes Completed):
Effective Manipulation Index (EMI): 0.2467  (Retention: 92.0% x Closure: 26.8%)
Success Rate (<= 10% error):    24.0% (6/25)
Mean Rotational Closure:             26.8%
Drop Rate:                            8.0% (2/25)
Mean Episode Steps:                 333.2 steps (6.66s)
Mean Angular Error:                 24.23° +/- 24.19°
======================================================================


```


## Stage 1D (2026-10-12 12:33:03)

```
 Rotational Closure:  72.3% | Status: SUCCESS (<= 4.1°)

[Episode 22/25] Object: 3 | Initial Target Angle:  41.4° | Success Threshold (10%): <= 4.14° | Horizon: 386 steps (7.72s)
  Finished: Steps: 500/386 | Final Error: 24.20° | Rotational Closure:  41.6% | Status: SUCCESS (<= 4.1°)

[Episode 23/25] Object: 2 | Initial Target Angle:  20.1° | Success Threshold (10%): <= 2.01° | Horizon: 270 steps (5.40s)
  Finished: Steps: 270/270 | Final Error:  9.16° | Rotational Closure:  54.4% | Status: TIMEOUT (9.2° > 2.0°)

[Episode 24/25] Object: 2 | Initial Target Angle:  42.5° | Success Threshold (10%): <= 4.25° | Horizon: 393 steps (7.86s)
  Finished: Steps: 500/393 | Final Error:  7.41° | Rotational Closure:  82.5% | Status: SUCCESS (<= 4.2°)

[Episode 25/25] Object: 3 | Initial Target Angle:  30.9° | Success Threshold (10%): <= 3.09° | Horizon: 324 steps (6.48s)
  Finished: Steps: 500/324 | Final Error: 12.75° | Rotational Closure:  58.7% | Status: SUCCESS (<= 3.1°)

======================================================================
STAGE 1 BENCHMARK SUMMARY (25 Episodes Completed):
Effective Manipulation Index (EMI): 0.2296  (Retention: 72.0% x Closure: 31.9%)
Success Rate (<= 10% error):    76.0% (19/25)
Mean Rotational Closure:             31.9%
Drop Rate:                           28.0% (7/25)
Mean Episode Steps:                 416.4 steps (8.33s)
Mean Angular Error:                 45.92° +/- 49.38°
======================================================================


```


## Stage 1 Final Benchmark (50 Episodes) (2026-10-12 12:34:13)

```
: 360/360 | Final Error: 46.42° | Rotational Closure:   0.0% | Status: TIMEOUT (46.4° > 3.7°)

[Episode 45/50] Object: 1 | Initial Target Angle:  41.0° | Success Threshold (10%): <= 4.10° | Horizon: 383 steps (7.66s)
  Finished: Steps: 235/383 | Final Error: 135.83° | Rotational Closure:   0.0% | Status: SUCCESS (<= 4.1°)

[Episode 46/50] Object: 1 | Initial Target Angle:  33.7° | Success Threshold (10%): <= 3.37° | Horizon: 340 steps (6.80s)
  Finished: Steps: 262/340 | Final Error: 139.67° | Rotational Closure:   0.0% | Status: SUCCESS (<= 3.4°)

[Episode 47/50] Object: 1 | Initial Target Angle:  11.4° | Success Threshold (10%): <= 1.14° | Horizon: 235 steps (4.70s)
  Finished: Steps: 235/235 | Final Error:  6.71° | Rotational Closure:  40.9% | Status: TIMEOUT (6.7° > 1.1°)

[Episode 48/50] Object: 1 | Initial Target Angle:  10.3° | Success Threshold (10%): <= 1.03° | Horizon: 231 steps (4.62s)
  Finished: Steps: 242/231 | Final Error: 118.86° | Rotational Closure:   0.0% | Status: SUCCESS (<= 1.0°)

[Episode 49/50] Object: 3 | Initial Target Angle:  14.7° | Success Threshold (10%): <= 1.47° | Horizon: 248 steps (4.96s)
  Finished: Steps: 500/248 | Final Error: 32.89° | Rotational Closure:   0.0% | Status: SUCCESS (<= 1.5°)

[Episode 50/50] Object: 3 | Initial Target Angle:  35.4° | Success Threshold (10%): <= 3.54° | Horizon: 350 steps (7.00s)
  Finished: Steps: 500/350 | Final Error: 11.56° | Rotational Closure:  67.4% | Status: SUCCESS (<= 3.5°)

======================================================================
STAGE 1 BENCHMARK SUMMARY (50 Episodes Completed):
Effective Manipulation Index (EMI): 0.1549  (Retention: 68.0% x Closure: 22.8%)
Success Rate (<= 10% error):    76.0% (38/50)
Mean Rotational Closure:             22.8%
Drop Rate:                           32.0% (16/50)
Mean Episode Steps:                 401.5 steps (8.03s)
Mean Angular Error:                 55.77° +/- 55.55°
======================================================================


```


## Stage 1D (Part 2 - 10k Consolidation) (2026-10-12 15:41:17)

```
inal Error: 167.89° | Rotational Closure:   0.0% | Status: DROPPED

[Episode 22/25] Object: 1 | Initial Target Angle:  25.4° | Success Threshold (10%): <= 2.54° | Horizon: 295 steps (5.90s)
  Finished: Steps: 266/295 | Final Error: 117.25° | Rotational Closure:   0.0% | Status: SUCCESS (<= 2.5°)

[Episode 23/25] Object: 3 | Initial Target Angle:  36.7° | Success Threshold (10%): <= 3.67° | Horizon: 357 steps (7.14s)
  Finished: Steps: 500/357 | Final Error: 11.08° | Rotational Closure:  69.8% | Status: SUCCESS (<= 3.7°)

[Episode 24/25] Object: 1 | Initial Target Angle:  43.3° | Success Threshold (10%): <= 4.33° | Horizon: 398 steps (7.96s)
  Finished: Steps: 236/398 | Final Error: 130.58° | Rotational Closure:   0.0% | Status: DROPPED

[Episode 25/25] Object: 2 | Initial Target Angle:  17.4° | Success Threshold (10%): <= 1.74° | Horizon: 259 steps (5.18s)
  Finished: Steps: 500/259 | Final Error: 19.71° | Rotational Closure:   0.0% | Status: SUCCESS (<= 1.7°)

======================================================================
STAGE 1 BENCHMARK SUMMARY (25 Episodes Completed):
Effective Manipulation Index (EMI): 0.1718  (Retention: 60.0% x Closure: 28.6%)
Success Rate (<= 10% error):    80.0% (20/25)
Mean Rotational Closure:             28.6%
Drop Rate:                           40.0% (10/25)
Mean Episode Steps:                 407.4 steps (8.15s)
Mean Angular Error:                 66.49° +/- 64.30°
======================================================================


```


## Stage 1E (5% Precision Lock-In) (2026-10-12 18:59:57)

```
 | Rotational Closure:   0.0% | Status: SUCCESS (<= 2.3°)

[Episode 22/25] Object: 3 | Initial Target Angle:  11.2° | Success Threshold (10%): <= 1.12° | Horizon: 234 steps (4.68s)
  Finished: Steps: 384/234 | Final Error: 19.38° | Rotational Closure:   0.0% | Status: SUCCESS (<= 1.1°)

[Episode 23/25] Object: 2 | Initial Target Angle:  18.0° | Success Threshold (10%): <= 1.80° | Horizon: 261 steps (5.22s)
  Finished: Steps: 500/261 | Final Error: 20.44° | Rotational Closure:   0.0% | Status: SUCCESS (<= 1.8°)

[Episode 24/25] Object: 1 | Initial Target Angle:  21.9° | Success Threshold (10%): <= 2.19° | Horizon: 279 steps (5.58s)
  Finished: Steps: 374/279 | Final Error: 119.61° | Rotational Closure:   0.0% | Status: SUCCESS (<= 2.2°)

[Episode 25/25] Object: 3 | Initial Target Angle:  13.5° | Success Threshold (10%): <= 1.35° | Horizon: 243 steps (4.86s)
  Finished: Steps: 402/243 | Final Error: 149.39° | Rotational Closure:   0.0% | Status: SUCCESS (<= 1.4°)

======================================================================
STAGE 1 BENCHMARK SUMMARY (25 Episodes Completed):
Effective Manipulation Index (EMI): 0.2221  (Retention: 76.0% x Closure: 29.2%)
Success Rate (<= 10% error):    88.0% (22/25)
Mean Rotational Closure:             29.2%
Drop Rate:                           24.0% (6/25)
Mean Episode Steps:                 432.6 steps (8.65s)
Mean Angular Error:                 40.78° +/- 43.41°
======================================================================


```


## Stage 1 Grand Final Benchmark (50 Episodes, Sub-2.5° Tolerance) (2026-10-12 19:01:06)

```
teps: 384/370 | Final Error: 161.64° | Rotational Closure:   0.0% | Status: SUCCESS (<= 3.9°)

[Episode 45/50] Object: 2 | Initial Target Angle:  15.3° | Success Threshold (10%): <= 1.53° | Horizon: 250 steps (5.00s)
  Finished: Steps: 500/250 | Final Error: 25.05° | Rotational Closure:   0.0% | Status: SUCCESS (<= 1.5°)

[Episode 46/50] Object: 1 | Initial Target Angle:  10.9° | Success Threshold (10%): <= 1.09° | Horizon: 234 steps (4.68s)
  Finished: Steps: 297/234 | Final Error: 111.41° | Rotational Closure:   0.0% | Status: SUCCESS (<= 1.1°)

[Episode 47/50] Object: 2 | Initial Target Angle:  40.6° | Success Threshold (10%): <= 4.06° | Horizon: 381 steps (7.62s)
  Finished: Steps: 381/381 | Final Error: 13.55° | Rotational Closure:  66.6% | Status: TIMEOUT (13.6° > 4.1°)

[Episode 48/50] Object: 2 | Initial Target Angle:  43.6° | Success Threshold (10%): <= 4.36° | Horizon: 400 steps (8.00s)
  Finished: Steps: 500/400 | Final Error:  5.57° | Rotational Closure:  87.2% | Status: SUCCESS (<= 4.4°)

[Episode 49/50] Object: 3 | Initial Target Angle:  21.3° | Success Threshold (10%): <= 2.13° | Horizon: 276 steps (5.52s)
  Finished: Steps: 276/276 | Final Error: 151.81° | Rotational Closure:   0.0% | Status: SUCCESS (<= 2.1°)

[Episode 50/50] Object: 1 | Initial Target Angle:  23.7° | Success Threshold (10%): <= 2.37° | Horizon: 287 steps (5.74s)
  Finished: Steps: 259/287 | Final Error: 99.24° | Rotational Closure:   0.0% | Status: SUCCESS (<= 2.4°)

======================================================================
STAGE 1 BENCHMARK SUMMARY (50 Episodes Completed):
Effective Manipulation Index (EMI): 0.1307  (Retention: 72.0% x Closure: 18.1%)
Success Rate (<= 10% error):    82.0% (41/50)
Mean Rotational Closure:             18.1%
Drop Rate:                           28.0% (14/50)
Mean Episode Steps:                 410.9 steps (8.22s)
Mean Angular Error:                 49.50° +/- 49.30°
======================================================================


```


## Stage 2C (18% Tolerance Adaptation) (2026-10-14 15:29:14)

```
|   69.9%  |   90.0%  |   10.0%  |   20.0%  |  42.8°     | 454.3
---------------------------------------------------------------------------------------------------------
RANGE 2 TOTAL | 60       |   36.7%  |   53.8%  |   75.0%  |   25.0%  |   36.7%  |  62.9°     | 499.0
  --> Range 2 Effective Manipulation Index (EMI): 0.4032 | Benchmark Score: 0.4946

======================================================================================
SECTION 3: GRAND COMPOSITE VERIFICATION SUMMARY (FULL 10° TO 90° ENVELOPE)
======================================================================================
Evaluation Metric                   | Range 1 (10°-40°)    | Range 2 (40°-90°)    | Grand Total    
------------------------------------------------------------------------------------------------
Total Evaluated Episodes            | 60                   | 60                   | 120            
Success Rate (@10% Error)           | 93.3               % | 36.7               % | 65.0          %
Mean Rotational Closure             | 94.2               % | 53.8               % | 74.0          %
Pre-Goal Retention Rate             | 96.7               % | 75.0               % | 85.8          %
Pre-Goal Drop Rate                  | 3.3                % | 25.0               % | 14.2          %
Raw Drop Rate (All Steps)           | 20.0               % | 36.7               % | 28.3          %
Mean Final Geodesic Error           | 38.4               ° | 62.9               ° | 50.6          °
Mean Steps per Episode              | 305.1                | 499.0                | 402.1          
Peak Hardware Torque                | 4.97             N·m | 6.55             N·m | 6.55        N·m
Effective Manipulation Index (EMI)  | 0.9108               | 0.4032               | 0.6351         
Composite Benchmark Score           | 0.9427               | 0.4946               | 0.7186         
================================================================================================


```


## Stage 2D (15% Tolerance Consolidation) (2026-10-14 16:54:31)

```
|   70.0%  |   90.0%  |   10.0%  |   20.0%  |  42.8°     | 454.3
---------------------------------------------------------------------------------------------------------
RANGE 2 TOTAL | 60       |   36.7%  |   53.8%  |   75.0%  |   25.0%  |   36.7%  |  62.5°     | 498.0
  --> Range 2 Effective Manipulation Index (EMI): 0.4036 | Benchmark Score: 0.4948

======================================================================================
SECTION 3: GRAND COMPOSITE VERIFICATION SUMMARY (FULL 10° TO 90° ENVELOPE)
======================================================================================
Evaluation Metric                   | Range 1 (10°-40°)    | Range 2 (40°-90°)    | Grand Total    
------------------------------------------------------------------------------------------------
Total Evaluated Episodes            | 60                   | 60                   | 120            
Success Rate (@10% Error)           | 93.3               % | 36.7               % | 65.0          %
Mean Rotational Closure             | 94.2               % | 53.8               % | 74.0          %
Pre-Goal Retention Rate             | 96.7               % | 75.0               % | 85.8          %
Pre-Goal Drop Rate                  | 3.3                % | 25.0               % | 14.2          %
Raw Drop Rate (All Steps)           | 20.0               % | 36.7               % | 28.3          %
Mean Final Geodesic Error           | 39.3               ° | 62.5               ° | 50.9          °
Mean Steps per Episode              | 305.5                | 498.0                | 401.7          
Peak Hardware Torque                | 4.97             N·m | 6.55             N·m | 6.55        N·m
Effective Manipulation Index (EMI)  | 0.9108               | 0.4036               | 0.6353         
Composite Benchmark Score           | 0.9427               | 0.4948               | 0.7187         
================================================================================================


```


## Stage 2E (12% Tolerance High Precision) (2026-10-14 18:36:18)

```
|   69.9%  |   90.0%  |   10.0%  |   20.0%  |  42.8°     | 454.3
---------------------------------------------------------------------------------------------------------
RANGE 2 TOTAL | 60       |   36.7%  |   53.7%  |   75.0%  |   25.0%  |   36.7%  |  62.9°     | 499.2
  --> Range 2 Effective Manipulation Index (EMI): 0.4030 | Benchmark Score: 0.4945

======================================================================================
SECTION 3: GRAND COMPOSITE VERIFICATION SUMMARY (FULL 10° TO 90° ENVELOPE)
======================================================================================
Evaluation Metric                   | Range 1 (10°-40°)    | Range 2 (40°-90°)    | Grand Total    
------------------------------------------------------------------------------------------------
Total Evaluated Episodes            | 60                   | 60                   | 120            
Success Rate (@10% Error)           | 93.3               % | 36.7               % | 65.0          %
Mean Rotational Closure             | 94.2               % | 53.7               % | 74.0          %
Pre-Goal Retention Rate             | 96.7               % | 75.0               % | 85.8          %
Pre-Goal Drop Rate                  | 3.3                % | 25.0               % | 14.2          %
Raw Drop Rate (All Steps)           | 20.0               % | 36.7               % | 28.3          %
Mean Final Geodesic Error           | 39.3               ° | 62.9               ° | 51.1          °
Mean Steps per Episode              | 305.5                | 499.2                | 402.4          
Peak Hardware Torque                | 4.97             N·m | 6.55             N·m | 6.55        N·m
Effective Manipulation Index (EMI)  | 0.9108               | 0.4030               | 0.6350         
Composite Benchmark Score           | 0.9427               | 0.4945               | 0.7186         
================================================================================================


```


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


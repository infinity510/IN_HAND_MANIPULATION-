# Best Champion Model (Stage 2 Large-Angle In-Hand Manipulation)

- **Checkpoint File**: `best_model.pth`
- **Source**: Iteration 20,700 
- **MD5 Hash**: `eb5188bea9f3b0bd7164d9b53b7a2ea0`
- **Target Precision Envelope**: 10.0% Relative Error Tolerance ($\epsilon_{\text{ratio}} = 0.10$), $90^\circ$ Large-Angle Envelope ($1.57$ rad)
- **Obs Dimension**: 39
- **Act Dimension**: 12 (Hardware-enforced joint limits $\le 120^\circ/100^\circ$, max torque $\le 10\text{ N}\cdot\text{m}$)

---

### Empirical Verification Performance (60 Randomized Out-of-Distribution Episodes)

| Object Geometry | Success Rate (@10% Error) | Rotational Closure | Pre-Goal Retention | Pre-Goal Drop Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Cylinder** | **$70.0\%$** | **$83.2\%$** | **$95.0\%$** | $5.0\%$ |
| **Cube** | **$60.0\%$** | **$80.2\%$** | **$100.0\%$** | $0.0\%$ |
| **Cuboid** | **$60.0\%$** | **$64.5\%$** | **$70.0\%$** | $30.0\%$ |
| **OVERALL COMPOSITE** | **$63.33\%$** | **$76.01\%$** | **$88.33\%$** | **$11.67\%$** |

- **Composite Benchmark Score**: **$0.4923$**

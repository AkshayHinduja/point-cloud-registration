# Point Cloud Registration  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://badge.fury.io/py/point-cloud-registration.svg?cache=1)](https://pypi.org/project/point-cloud-registration/) ![Python application](https://github.com/scomup/point-cloud-registration/actions/workflows/python-app.yml/badge.svg)

`point-cloud-registration` is a **pure Python**, **lightweight**, and **fast** point cloud registration library.  
It outperforms PCL and Open3D's registration in speed while relying **only on NumPy** for computations.

## Features  
✅ **Pure Python** – No compiled extensions, works everywhere  
✅ **Fast & Lightweight** – Optimized algorithms with minimal overhead  
✅ **NumPy-based API** – Seamless integration with scientific computing workflows  

The following registration algorithms are supported, with our **pure Python** `point-cloud-registration` being **even faster than the C++** versions of PCL and Open3D (C++ & Python wrappers).

### Current Support Algorithm and Speed Comparison [^1]

| Method                          | Our (sec) | Open3D (sec) | PCL (sec) |
|---------------------------------|-----------|--------------|-----------|
| Point-to-Point ICP              | **0.502** | 1.511        | 0.931      |
| Point-to-Plane ICP [^2]         | **0.334** | 0.677        | 0.835      |
| Voxelized Point-to-Plane ICP    | **0.420** | N/A          | N/A       |
| Normal Distributions Transform (NDT) | **0.511** | N/A          | 1.782      |
| Normal Estimation               | 2.201    | **1.708**    | 2.048      |

---

[^1]: **Note**: The above times are based on the test data in `data/B-01.pcd` with over 1,000,000 points, licensed under CC BY 4.0. For more information, see `data/README.md`.  

## Installation  

Install via pip:  

```bash
pip install point-cloud-registration
pip install q3dviewer==1.1.6 # (optional) for visual demo
```

You can benchmark it yourself using the following commands:
```bash
cd benchmark
python3 speed_test_comparison.py # Compare our implementation with Open3D
python3 speed_test_comparison_mkl.py # (optional) you can further boost it by using mkl numpy
mkdir build && cd build
cmake .. && make
./speed_test_comparison # Compare with PCL
```

[^2]: Without Normal Estimation

## Usage  

```python
#!/usr/bin/env python3

import numpy as np
from point_cloud_registration import ICP, PlaneICP, NDT, VPlaneICP

# Example point clouds
target = np.random.rand(100, 3)  # Nx3 point numpy array
scan = np.random.rand(80, 3)    # Mx3 point numpy array

icp = VPlaneICP(voxel_size=0.5, max_iter=30, max_dist=2, tol=1e-3)
icp.set_target(target)  # Set the target point cloud
T_new = icp.align(scan, init_T=np.eye(4))  # Fit the scan to the target
print("Estimated Transform matrix:\n", T_new)
```

## Roadmap  
🚀 **Upcoming Features & Enhancements**:  
- [x] **Point-to-Point ICP** – Basic ICP implementation  
- [x] **Point-to-Plane ICP** – Improved accuracy using normal constraints  
- [ ] **Generalized ICP (GICP)** – Handles anisotropic noise and improves robustness  
- [x] **Normal Distributions Transform (NDT)** – Grid-based registration for high-noise environments  
- [ ] **Further optimizations** while staying pure Python  
### Demo

Explore the capabilities of the library with the following demos:

#### Visualize Estimated Normals

Quickly estimate and visualize normals using our fast Python implementation:

```bash
python3 demo_estimate_normals.py
```

![Estimated Normals](imgs/norm.png)

#### Visualize Voxels

Visualize 3D voxelization with our efficient Python-based algorithm:

```bash
python3 demo_visualize_voxels.py
```

![Voxel Visualization](imgs/voxel.png)

#### Try Point Cloud Registration

Explore our Cloud Registration algorithms using a sample GUI. You can experiment with different settings:

```bash
python3 demo_matching.py
```

![demo](imgs/demo.png)

### Comparison of Registration Methods

| Method                        | Objective Function*                                         | Data Representation   | Speed         | Precision    |
|-------------------------------|-----------------------------------------------------------|------------------------|---------------|--------------|
| Point-to-Point ICP            | $\sum \| T p_i - q_i \|^2$                                | Point-Based            | Fast          | Moderate     |
| Point-to-Plane ICP            | $\sum \| n_i^T (T p_i - q_i) \|^2$                        | Point-Based (with normals) | Fast | High | 
| Voxelized Point-to-Plane ICP  | $\sum \| n_i^T (T p_i - q_i) \|^2$                        | Voxel-Based (with normals) | Very Fast | High | 
| Generalized ICP (GICP)        | $\sum (T p_i - q_i)^T (C_i^Q + R C_i^P R^T)^{-1} (T p_i - q_i)$ | Point-Based (with covariances) | Moderate | Very High | 
| Normal Distributions Transform (NDT) | $\sum (T p_i - \mu_i)^T \Sigma_i^{-1} (T p_i - \mu_i)$ | Voxel-Based (with covariances) | Very Fast | Moderate |

## Degeneracy handling

When the scene is geometrically under-determined — a single flat wall, a straight corridor, a surface of revolution — the 6×6 Hessian is rank-deficient or ill conditioned and the plain Gauss-Newton step slides along the unconstrained directions. `align()` offers three independent mitigation families (all **off by default**; the default path is untouched plain Gauss-Newton):

- **`use_solution_remapping=True`** (+ `sr_lambda_threshold`) — degeneracy-aware ICP per Hinduja, Ho & Kaess, *Degeneracy-Aware Factors with Applications to Underwater SLAM*, IROS 2019 ([doi:10.1109/IROS40897.2019.8968577](https://doi.org/10.1109/IROS40897.2019.8968577)), which introduced solution remapping inside the ICP iteration; the remapping update originates from Zhang, Kaess & Singh, *On Degeneracy of Optimization-based State Estimation Problems*, ICRA 2016. Eigen-analyses the 6×6 Hessian and projects the update so its component along every degenerate eigen-direction is zero. Pair it with `lm_damping` when the raw Hessian may be exactly singular. Note the published criterion compares an eigenvalue against `sqrt(λ_max/λ_min)`, so it is *not* invariant to a uniform rescaling of `H`; supply `sr_lambda_threshold` if you have a calibrated curvature floor.
- **`lm_damping=True`** — trace-scaled Levenberg–Marquardt damping (`H + λI`, `λ = 1e-4·tr(H)/6`). Keeps the linear solve well posed where the plain one raises `LinAlgError`. It does not move the minimum, only the step length, so weak directions converge slowly rather than being suppressed.
- **`dcreg_mode='pcg' | 'clamped'`** (+ `dcreg_kappa_threshold`, `dcreg_kappa_target`, `dcreg_pcg_tolerance`, `dcreg_pcg_max_iterations`) — decoupled Schur-complement analysis per Hu et al., *DCReg*, IJRR 2026 ([arXiv:2509.06285](https://arxiv.org/abs/2509.06285)); independent implementation from the published mathematics. Detection is per *physical axis* per block and tests a ratio against a ratio, so it is scale invariant and immune to the lever-arm inflation of the rotation block. The two solves differ fundamentally: **`'pcg'`** is a preconditioned CG on the *unmodified* normal equations and therefore returns the plain Gauss-Newton step at convergence — it buys conditioning, not mitigation; **`'clamped'`** adds the per-block spectral deficit along the flagged axes and *does* move the minimum, which means `dcreg_kappa_threshold` decides how much recoverable signal you discard. `dcreg_mode` and `use_solution_remapping` are mutually exclusive.

`analyse_hessian_decoupled(H, kappa_threshold=10.0)` is the standalone detection API — usable without any solver change, e.g. to gate or covariance-weight a registration result downstream.

```python
from point_cloud_registration import PlaneICP, analyse_hessian_decoupled

icp = PlaneICP(max_iter=30, max_dist=2.0, tol=1e-6)
icp.set_target(target)
T = icp.align(source, dcreg_mode='clamped', dcreg_kappa_threshold=10.0)

deg = analyse_hessian_decoupled(icp.last_hessian, kappa_threshold=10.0)
print(deg.degenerate_mask)   # [tx ty tz wx wy wz]
print(deg.cond_schur_t, deg.cond_schur_R)
```

`tests/test_mitigation_ab.py` measures all of the above against each other on shared fixtures and documents what each one costs; run it with `pytest tests/test_mitigation_ab.py -s -k summary_table` for the comparison table.

---

## License  

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

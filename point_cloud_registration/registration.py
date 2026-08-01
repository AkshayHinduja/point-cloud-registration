"""
Copyright 2025 Liu Yang
Distributed under MIT license. See LICENSE for more information.
"""

import numpy as np
from point_cloud_registration.math_tools import plus
from point_cloud_registration.degeneracy import (
    analyse_hessian,
    apply_sr_solve,
    analyse_hessian_decoupled,
    dcreg_solve,
)


class Registration:
    def __init__(self, max_iter=30, tol=1e-3):
        """
        Base class for point cloud registration methods.
        :param max_iter: Maximum number of iterations.
        :param tol: Convergence tolerance.
        """
        self.max_iter = max_iter
        self.tol = tol
        self._is_target_set = False

    def is_target_set(self):
        """
        Check if the target point cloud is set.
        :return: True if target is set, False otherwise.
        """
        return self._is_target_set

    def set_target(self, target):
        """
        Set the target point cloud.
        :param target: Target point cloud (Nx3 array).
        """
        self._is_target_set = True
        raise NotImplementedError("set_target is not implemented.")

    def update_target(self, target):
        """
        Update the target map continuously.
        It is very useful for saving time by avoiding rebuild the KDTree or 3D voxels.
        But we do not plan to implement it currently.
        :param target: Target point cloud (Nx3 array).
        """
        raise NotImplementedError("update_target is not implemented.")

    def linearize(self, cur_T, source):
        """
        Linearize the objective function and compute the Jacobian and residual.
        :param
        cur_T: Current transformation (4x4 array).
        source: Source point cloud (Nx3 array).
        :return: Jacobian (Nx6 array), residual (Nx3 array), weights (N array).
        """
        raise NotImplementedError("linearize is not implemented.")
    
    def calc_H_g_e2(self, cur_T, source, dx_norm=np.inf):
        """
        Compute the Hessian, gradient, and squared error.
        :param cur_T: Current transformation (4x4 array).
        source: Source point cloud (Nx3 array).
        :return: Hessian (6x6 array), gradient (6 array), squared error (scalar).
        """
        Js, rs, weights = self.linearize(cur_T, source)
        # use einsum to parallelize the computation
        JsT = Js.transpose(0, 2, 1)
        H = np.einsum('nij,njk,n->ik', JsT, Js, weights)
        g = np.einsum('nij,nj,n->i', JsT, rs, weights)
        e2 = np.einsum('ni,ni,n->', rs, rs, weights)
        return H, g, e2


    def align(self, source, init_T=np.eye(4), verbose=False,
              use_solution_remapping=False, sr_lambda_threshold=None,
              lm_damping=False,
              dcreg_mode=None, dcreg_kappa_threshold=10.0, dcreg_kappa_target=None,
              dcreg_pcg_tolerance=1e-6, dcreg_pcg_max_iterations=10):
        """
        Gauss-Newton ICP alignment.

        :param source: Source point cloud (Nx3 array).
        :param init_T: Initial transformation (4x4 array).
        :param verbose: Print error at each iteration.
        :param use_solution_remapping: If True, zero the step in degenerate Hessian
            eigenvector directions at each iteration (SR mode).
        :param sr_lambda_threshold: Override the condition-number threshold for SR.
            None = adaptive (sqrt(lambda_max / lambda_min)).
        :param lm_damping: If True, solve the damped system (H + lambda*I) dx = -g
            instead of H dx = -g, with lambda scaled to the trace of H
            (Levenberg-Marquardt). This keeps the linear solve well posed on
            geometry that leaves H singular or near-singular — a single flat
            surface, a straight corridor, too few correspondences — where the
            plain solve raises numpy.linalg.LinAlgError or returns a step
            dominated by noise. On well-conditioned data the damping is small
            enough to leave the solution unchanged. Pair it with
            use_solution_remapping when the Hessian may be exactly rank
            deficient: the degeneracy analysis runs on the raw H so the
            classification stays honest, while the solve uses the damped one.
        :param dcreg_mode: None (default), 'pcg' or 'clamped'. Enables the
            DCReg-style decoupled mitigation solve (Hu et al., IJRR 2026,
            arXiv:2509.06285) in place of the plain Gauss-Newton one. Each
            iteration analyses the RAW H per block via Schur complements and
            solves through degeneracy.dcreg_solve. Read that function's
            docstring before choosing a mode — in particular, 'pcg' solves the
            unmodified normal equations and so returns the plain Gauss-Newton
            step at full convergence; it improves conditioning, it does not move
            the minimum. 'clamped' does move it, along the flagged axes only.
            Mutually exclusive with use_solution_remapping: both are mitigation
            strategies for the same problem and stacking them is described by
            neither paper.
        :param dcreg_kappa_threshold: Per-block eigenvalue-ratio threshold above
            which an axis is called degenerate (passed through to
            analyse_hessian_decoupled).
        :param dcreg_kappa_target: Target ratio used when clamping the flagged
            eigenvalues. None = same as dcreg_kappa_threshold.
        :param dcreg_pcg_tolerance: Convergence tolerance for dcreg_mode='pcg'.
        :param dcreg_pcg_max_iterations: Iteration budget for dcreg_mode='pcg'
            (floored at 6 inside the solver).
        :return: Final transformation (4x4 array).
        """
        if dcreg_mode is not None and dcreg_mode not in ('pcg', 'clamped'):
            raise ValueError(
                f"Unknown dcreg_mode {dcreg_mode!r}; expected None, 'pcg' or 'clamped'"
            )
        if dcreg_mode is not None and use_solution_remapping:
            raise ValueError(
                "dcreg_mode and use_solution_remapping are mutually exclusive "
                "mitigation strategies; enable at most one."
            )

        if self.is_target_set() is False:
            raise ValueError("Target is not set.")

        source = source.astype(np.float32)
        cur_T = init_T
        H_final = None
        converged = False

        for i in range(self.max_iter):
            H, g, e2 = self.calc_H_g_e2(cur_T, source)
            H_final = H

            if verbose:
                print(f"iter {i}, error {e2}")

            if lm_damping:
                trace_H = np.trace(H)
                lambda_lm = max(1e-4 * trace_H / 6.0 if trace_H > 0 else 1e-3, 1e-6)
                H_solve = H + lambda_lm * np.eye(6)
            else:
                H_solve = H

            if use_solution_remapping:
                # Eigendecompose the raw H (correct degeneracy thresholding) but
                # solve on H_solve (numerical stability): an isotropic +lambda*I
                # shift leaves the eigenvectors identical.
                deg = analyse_hessian(H.astype(float), lambda_threshold=sr_lambda_threshold)
                dx = apply_sr_solve(H_solve.astype(float), g.astype(float), deg)
            elif dcreg_mode is not None:
                # Same composition rule as the SR path above: characterize the
                # RAW H (damping would inflate the small eigenvalues before the
                # ratio test sees them, under-reporting degeneracy) but solve
                # H_solve.  Note that combining lm_damping with dcreg_mode is an
                # extension beyond the paper, which damps nothing and relies on
                # the clamped spectrum alone; it is off by default.
                deg = analyse_hessian_decoupled(
                    H.astype(float),
                    kappa_threshold=dcreg_kappa_threshold,
                    kappa_target=dcreg_kappa_target,
                )
                dx = dcreg_solve(
                    H_solve.astype(float), g.astype(float), deg,
                    mode=dcreg_mode,
                    pcg_tolerance=dcreg_pcg_tolerance,
                    pcg_max_iterations=dcreg_pcg_max_iterations,
                )[0]
            elif lm_damping:
                try:
                    dx = -np.linalg.solve(H_solve, g)
                except np.linalg.LinAlgError:
                    dx = -np.linalg.lstsq(H_solve, g, rcond=None)[0]
            else:
                dx = -np.linalg.solve(H, g)

            dx_norm = np.linalg.norm(dx)
            if dx_norm < self.tol:
                converged = True
                break

            cur_T = plus(cur_T, dx)

        # If max_iter was exhausted, cur_T advanced past the last H computation — recompute.
        if not converged and H_final is not None:
            H_final, _, _ = self.calc_H_g_e2(cur_T, source)
        self._last_hessian = H_final
        return cur_T

    @property
    def last_hessian(self):
        """Return the 6×6 Hessian from the most recent align() call, or None."""
        return getattr(self, '_last_hessian', None)
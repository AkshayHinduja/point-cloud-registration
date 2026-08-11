"""
Copyright 2025 Liu Yang
Distributed under MIT license. See LICENSE for more information.
"""

import numpy as np
from point_cloud_registration.math_tools import plus


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
        self._last_hessian = None

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


    def align(self, source, init_T=np.eye(4), verbose=False):
        """
        use Gauss-Newton method to find the transformation 
        that aligns the source point cloud to the target point cloud.
        :param source: Source point cloud (Nx3 array).
        :param init_T: Initial transformation (4x4 array).
        :param verbose: Print error at each iteration.
        :return: Final transformation (4x4 array).
        """
        if self.is_target_set() is False:
            raise ValueError("Target is not set.")

        source = source.astype(np.float32)
        # Copy: align() must not return the caller's array (or the shared
        # mutable np.eye(4) default) when it converges before the first step.
        cur_T = init_T.copy()
        H_final = None
        converged = False

        for i in range(self.max_iter):
            H, g, e2 = self.calc_H_g_e2(cur_T, source)
            H_final = H
            if verbose:
                print(f"iter {i}, error {e2}")

            # solve the linear system
            dx = -np.linalg.solve(H, g)

            # check convergence
            dx_norm = np.linalg.norm(dx)
            if dx_norm < self.tol:
                converged = True
                break

            # Update transformation
            cur_T = plus(cur_T, dx)

        # If max_iter was exhausted, cur_T advanced past the last
        # linearization, so recompute the Hessian at the returned pose.
        if not converged and H_final is not None:
            H_final, _, _ = self.calc_H_g_e2(cur_T, source)
        self._last_hessian = H_final
        return cur_T

    @property
    def last_hessian(self):
        """
        The 6x6 Gauss-Newton Hessian (J^T W J) evaluated at the pose returned
        by the most recent align() call, or None before any align().

        It is expressed in the right-tangent (body) frame of that pose, DOF
        order [tx, ty, tz, wx, wy, wz] — the increment coordinates consumed by
        math_tools.plus().  Consumers that reason about world-frame directions
        (covariance extraction, observability analysis) must congruence-
        transform the translation block with the pose's rotation:
        H_world = blockdiag(R, I) @ H @ blockdiag(R, I).T after mapping
        dt_world = R @ dt_body.
        """
        return self._last_hessian
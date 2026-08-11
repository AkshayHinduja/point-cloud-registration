"""
Behavioral tests for Registration.align().
"""
import numpy as np
import pytest

from point_cloud_registration.icp import ICP
from point_cloud_registration.plane_icp import PlaneICP
from point_cloud_registration.math_tools import expSO3, makeT


@pytest.fixture
def well_conditioned_pair():
    """
    Three mutually orthogonal planes ("corner") — normals span R^3, so all six
    DOF are constrained and the Hessian is well conditioned.

    Returns (target, source, T_true) where T_true is the transform align()
    should recover, i.e. the one that maps source onto target.
    """
    rng = np.random.default_rng(0)
    n = 400
    a = rng.uniform(0.0, 4.0, size=(n, 2))
    b = rng.uniform(0.0, 4.0, size=(n, 2))
    c = rng.uniform(0.0, 4.0, size=(n, 2))
    face_z = np.column_stack([a[:, 0], a[:, 1], np.zeros(n)])
    face_x = np.column_stack([np.zeros(n), b[:, 0], b[:, 1]])
    face_y = np.column_stack([c[:, 0], np.zeros(n), c[:, 1]])
    target = np.vstack([face_z, face_x, face_y])

    R = expSO3(np.array([0.02, -0.03, 0.05]))
    t = np.array([0.15, -0.1, 0.08])
    source = (R @ target.T).T + t
    T_true = np.linalg.inv(makeT(R, t))
    return target, source, T_true


def test_last_hessian_lifecycle(well_conditioned_pair):
    """last_hessian is None before any align and a 6x6 array afterwards."""
    target, source, _ = well_conditioned_pair
    engine = PlaneICP(max_iter=30, max_dist=2.0, tol=1e-6)
    engine.set_target(target)

    assert engine.last_hessian is None

    engine.align(source)

    H = engine.last_hessian
    assert isinstance(H, np.ndarray)
    assert H.shape == (6, 6)
    assert np.all(np.isfinite(H))


def test_last_hessian_recomputed_when_max_iter_exhausted(well_conditioned_pair):
    """
    When align() runs out of iterations, cur_T has advanced past the last
    linearization, so the stored Hessian must be recomputed at the returned
    pose rather than left at the pre-step one.
    """
    target, source, _ = well_conditioned_pair
    engine = PlaneICP(max_iter=1, max_dist=2.0, tol=1e-6)
    engine.set_target(target)

    T = engine.align(source)

    H_at_returned = engine.calc_H_g_e2(T, source)[0]
    H_at_init = engine.calc_H_g_e2(np.eye(4), source)[0]

    assert np.allclose(engine.last_hessian, H_at_returned), (
        "last_hessian is not the Hessian at the returned pose; max diff "
        f"{np.max(np.abs(engine.last_hessian - H_at_returned))}"
    )
    assert not np.allclose(engine.last_hessian, H_at_init), (
        "last_hessian is stale (equals the pre-step Hessian)"
    )


def test_align_does_not_alias_init_T():
    """
    align() must return a transform the caller owns.

    With source == target the very first step is ~zero, so align()
    converges before ever calling plus(): without the defensive copy it
    returns the init_T object itself — and with the mutable np.eye(4)
    default, a caller mutating the result silently corrupts the default
    for every subsequent align() call in the process.
    """
    np.random.seed(1)
    target = np.random.rand(100, 3)
    icp = ICP(max_iter=10, max_dist=2.0, tol=1e-3)
    icp.set_target(target)
    source = target.astype(np.float32)

    init_T = np.eye(4)
    T = icp.align(source, init_T=init_T)
    assert T is not init_T

    # Mutating the result must not corrupt the shared default argument.
    T_default = icp.align(source)
    T_default[0, 3] = 123.0
    T_again = icp.align(source)
    np.testing.assert_allclose(T_again, np.eye(4), atol=1e-6)

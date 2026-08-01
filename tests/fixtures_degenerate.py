"""
Shared point-cloud pair builders for the degeneracy / mitigation tests.

Plain functions, not pytest fixtures, so that both ``test_registration_dcreg.py``
(which wraps them as fixtures) and ``test_mitigation_ab.py`` (which needs several
of them inside one parametrized test) can use the same geometry.  Every builder
is deterministic: fixed seeds and fixed grids, no wall-clock or platform inputs.

Each returns ``(target, source)`` — or ``(target, source, T_true)`` where a
ground-truth transform is meaningful — with ``source`` displaced from ``target``
by a known amount, so a test can ask "how much of that displacement did the
solver recover?".
"""
import numpy as np

from point_cloud_registration.math_tools import expSO3, makeT


def build_well_conditioned_pair():
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


def build_flat_plane_pair():
    """
    A perfectly flat z=0 grid as target, source the same grid shifted in +z.

    Every surface normal is (0, 0, 1), so the plane-ICP Jacobian rows are
    [0, 0, 1, y, -x, 0]: tx, ty and wz never appear and H is exactly rank 3.
    Those three columns of J are identically zero, so the matching entries of
    the gradient are exactly zero too — which is why several strategies produce
    *exactly* 0.0 in-plane rather than merely small numbers.

    Returns (target, source).  The true correction is (0, 0, -0.5).
    """
    g = np.arange(-5.0, 5.0 + 1e-9, 1.0)
    xx, yy = np.meshgrid(g, g)
    target = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    source = target + np.array([0.0, 0.0, 0.5])
    return target, source


def build_near_degenerate_pair():
    """
    A gently curved surface: ill conditioned but NOT rank deficient.

    The diagonal blocks of H are invertible, so the decoupled analysis
    succeeds; tx, ty and wz are flagged at kappa_threshold=10 and the
    translation Schur complement has a condition number around 3.3e5.

    The offset is deliberately LATERAL as well as vertical (0.6 m in x, 0.2 m
    in y, 0.5 m in z).  The curvature makes x and y genuinely recoverable —
    plain Gauss-Newton nails all three — so a strategy that fails to recover
    them is discarding signal the data supported, which is exactly the trade
    the A/B tests measure.

    Returns (target, source).  The true correction is (-0.6, -0.2, -0.5).
    """
    g = np.arange(-5.0, 5.0 + 1e-9, 0.5)
    xx, yy = np.meshgrid(g, g)
    zz = 0.05 * (xx ** 2 + 1.3 * yy ** 2) / 10.0
    target = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    source = target + np.array([0.6, 0.2, 0.5])
    return target, source


LEVER_ARM_OFFSET = 50.0
LEVER_ARM_ROLL = 0.01


def build_lever_arm_pair(roll=LEVER_ARM_ROLL, offset=LEVER_ARM_OFFSET):
    """
    The corner cloud, rigidly translated ``offset`` metres along +x, with the
    source rotated by ``roll`` radians about the cloud's own centroid x-axis.

    The geometry is locally identical to :func:`build_well_conditioned_pair` —
    the same three orthogonal faces, so the same local observability — but the
    Jacobian's rotation columns are ``p x n`` with ``|p| ~ 50``, which inflates
    the wy and wz curvature by roughly ``offset**2``.  wx gets no such boost:
    the offset is parallel to the x axis, so the extra lever arm about x is
    zero.  That is the scale disparity the decoupled analysis exists to strip
    out, manufactured on purpose.

    The perturbation is a pure rotation about the *centroid*, so the induced
    translation of the cloud's own frame is zero and "did the solver recover
    the roll?" is a well-posed question independent of the offset.

    Returns (target, source).
    """
    target, _, _ = build_well_conditioned_pair()
    target = target + np.array([offset, 0.0, 0.0])
    centroid = target.mean(axis=0)
    R = expSO3(np.array([roll, 0.0, 0.0]))
    source = (R @ (target - centroid).T).T + centroid
    return target, source

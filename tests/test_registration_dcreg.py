"""
Integration tests for the DCReg mitigation modes wired into
``Registration.align()`` via the ``dcreg_mode`` parameter.

Fixtures and style mirror ``tests/test_registration_align.py``.

The unit-level behaviour of the solver lives in ``tests/test_dcreg_solve.py``;
what is tested here is the *wiring*: parameter validation, that the default path
is untouched, and what each mode actually does to a real ICP loop on real
geometry.
"""

import numpy as np
import pytest

from point_cloud_registration.plane_icp import PlaneICP
from point_cloud_registration.math_tools import expSO3, makeT
from point_cloud_registration.degeneracy import analyse_hessian_decoupled


@pytest.fixture
def well_conditioned_pair():
    """
    Three mutually orthogonal planes ("corner") — normals span R^3, so all six
    DOF are constrained and the Hessian is well conditioned.  Identical to the
    fixture in ``test_registration_align.py``.

    Returns (target, source, T_true).
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


@pytest.fixture
def degenerate_pair():
    """
    A perfectly flat z=0 grid as target, source the same grid shifted in +z.

    Every surface normal is (0, 0, 1), so the Jacobian rows are
    [0, 0, 1, y, -x, 0]: tx, ty and wz never appear and H is exactly rank 3.
    """
    g = np.arange(-5.0, 5.0 + 1e-9, 1.0)
    xx, yy = np.meshgrid(g, g)
    target = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    source = target + np.array([0.0, 0.0, 0.5])
    return target, source


@pytest.fixture
def near_degenerate_pair():
    """
    A gently curved surface: ill conditioned but NOT rank deficient.

    This fixture exists because the flat-plane one cannot exercise the DCReg
    solvers at all — see :func:`test_flat_plane_gates_out_the_decoupled_analysis`.
    Here the diagonal blocks of H are invertible, the decoupled analysis
    succeeds, tx / ty / wz are flagged, and the translation Schur complement has
    a condition number around 3.3e5.  This is the fixture on which mode='pcg'
    actually runs its Krylov iteration.

    The offset is deliberately LATERAL as well as vertical (0.6 m in x, 0.2 m in
    y, 0.5 m in z).  A pure z offset would be useless as a discriminator: there
    would be no x/y correction for any solver to either recover or suppress, and
    all three modes would agree to within 3e-6 while telling us nothing.  The
    curvature of the surface does make x and y genuinely recoverable, which is
    what lets these tests show that 'clamped' suppresses a correction the data
    actually supported.
    """
    g = np.arange(-5.0, 5.0 + 1e-9, 0.5)
    xx, yy = np.meshgrid(g, g)
    zz = 0.05 * (xx ** 2 + 1.3 * yy ** 2) / 10.0
    target = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    source = target + np.array([0.6, 0.2, 0.5])
    return target, source


def _yaw(T):
    return float(np.arctan2(T[1, 0], T[0, 0]))


def _engine(target, **kw):
    e = PlaneICP(max_iter=kw.pop('max_iter', 30), max_dist=2.0, tol=1e-6)
    e.set_target(target)
    return e


# --------------------------------------------------------------------------
# 7 - Parameter validation
# --------------------------------------------------------------------------

class TestParameterValidation:
    def test_dcreg_and_solution_remapping_are_mutually_exclusive(self, well_conditioned_pair):
        """
        Both are mitigation strategies for the same problem and they compose
        into something neither paper describes, so combining them is rejected
        rather than silently resolved.
        """
        target, source, _ = well_conditioned_pair
        engine = _engine(target)
        with pytest.raises(ValueError, match="mutually exclusive"):
            engine.align(source, dcreg_mode='pcg', use_solution_remapping=True)

    def test_unknown_dcreg_mode_rejected(self, well_conditioned_pair):
        target, source, _ = well_conditioned_pair
        engine = _engine(target)
        with pytest.raises(ValueError, match="dcreg_mode"):
            engine.align(source, dcreg_mode='foo')

    def test_validation_happens_before_any_iteration(self, well_conditioned_pair):
        """A rejected configuration must not leave a half-updated Hessian behind."""
        target, source, _ = well_conditioned_pair
        engine = _engine(target)
        with pytest.raises(ValueError):
            engine.align(source, dcreg_mode='foo')
        assert engine.last_hessian is None


# --------------------------------------------------------------------------
# 8 - Well-conditioned geometry: both modes agree with the default solver
# --------------------------------------------------------------------------

class TestWellConditionedParity:
    def test_both_modes_match_the_default_solve(self, well_conditioned_pair):
        """
        On the corner fixture the decoupled analysis flags nothing, so 'clamped'
        adds a zero regularizer and 'pcg' preconditions a system it then solves
        exactly.  Both must reproduce the plain Gauss-Newton result.
        """
        target, source, _ = well_conditioned_pair

        T_default = _engine(target).align(source)
        T_pcg = _engine(target).align(source, dcreg_mode='pcg')
        T_clamped = _engine(target).align(source, dcreg_mode='clamped')

        assert np.allclose(T_default, T_pcg, atol=1e-3), (
            f"pcg diverged from default; max diff {np.max(np.abs(T_default - T_pcg))}"
        )
        assert np.allclose(T_default, T_clamped, atol=1e-3), (
            f"clamped diverged from default; max diff "
            f"{np.max(np.abs(T_default - T_clamped))}"
        )

    def test_fixture_really_is_unflagged(self, well_conditioned_pair):
        """Guards the premise of the test above."""
        target, source, _ = well_conditioned_pair
        engine = _engine(target)
        H = engine.calc_H_g_e2(np.eye(4), source.astype(np.float32))[0]
        deg = analyse_hessian_decoupled(H.astype(float), kappa_threshold=10.0)
        assert deg.factorization_ok
        assert not deg.is_degenerate


# --------------------------------------------------------------------------
# 9 - Degenerate geometry
# --------------------------------------------------------------------------

class TestFlatPlaneDegenerate:
    """
    The honest result on the flat plane: the DCReg *mathematics* never runs.

    H_tt is rank 1 and H_ww is rank 2, so neither Schur complement exists, the
    decoupled analysis returns ``factorization_ok=False``, and both modes take
    the ``lstsq`` fallback on the first iteration and every one after.  The
    excellent behaviour these tests observe — zero drift in x, y and yaw, z
    converging onto the plane — is the minimum-norm property of
    ``numpy.linalg.lstsq``, not a DCReg mitigation.  Recorded explicitly so the
    numbers are not later misread as evidence that the method works here.
    """

    def test_flat_plane_gates_out_the_decoupled_analysis(self, degenerate_pair):
        target, source = degenerate_pair
        engine = _engine(target)
        H = engine.calc_H_g_e2(np.eye(4), source.astype(np.float32))[0]

        eigenvalues = np.linalg.eigvalsh(H)
        assert np.count_nonzero(eigenvalues <= 0.0) == 3, f"eigenvalues: {eigenvalues}"

        deg = analyse_hessian_decoupled(H.astype(float), kappa_threshold=10.0)
        assert deg.factorization_ok is False, (
            "premise of this test class: the flat plane has no Schur complement"
        )

    def test_clamped_is_finite_with_bounded_drift(self, degenerate_pair):
        """
        Measured: x, y and yaw drift are exactly 0.0 and z lands on -0.5.  That
        is the lstsq minimum-norm step (the unobserved directions lie in the
        null space of H, and lstsq puts nothing there), not clamping.
        """
        target, source = degenerate_pair

        T = _engine(target).align(source, dcreg_mode='clamped')

        assert np.all(np.isfinite(T))
        assert abs(T[0, 3]) < 1e-6, f"x drift {T[0, 3]}"
        assert abs(T[1, 3]) < 1e-6, f"y drift {T[1, 3]}"
        assert abs(_yaw(T)) < 1e-6, f"yaw drift {_yaw(T)}"
        assert T[2, 3] == pytest.approx(-0.5, abs=1e-3), f"z did not converge: {T[2, 3]}"

    def test_pcg_is_finite_and_identical_to_clamped_here(self, degenerate_pair):
        """
        Because both modes bail to the same fallback before either solver runs,
        they return bit-identical transforms on this fixture.  If that ever
        stops being true, the fallback gate has changed.
        """
        target, source = degenerate_pair

        T_pcg = _engine(target).align(source, dcreg_mode='pcg')
        T_clamped = _engine(target).align(source, dcreg_mode='clamped')

        assert np.all(np.isfinite(T_pcg))
        np.testing.assert_array_equal(T_pcg, T_clamped)

    def test_no_exception_where_the_plain_solver_raises(self, degenerate_pair):
        """The plain Gauss-Newton path cannot even run on this geometry."""
        target, source = degenerate_pair
        with pytest.raises(np.linalg.LinAlgError):
            _engine(target).align(source)

        T = _engine(target).align(source, dcreg_mode='pcg')
        assert np.all(np.isfinite(T))


class TestNearDegenerate:
    """
    The fixture where mode='pcg' genuinely executes, so its real effect is
    observable rather than gated out.
    """

    def test_fixture_is_flagged_but_factorizable(self, near_degenerate_pair):
        target, source = near_degenerate_pair
        engine = _engine(target)
        H = engine.calc_H_g_e2(np.eye(4), source.astype(np.float32))[0]
        deg = analyse_hessian_decoupled(H.astype(float), kappa_threshold=10.0)

        assert deg.factorization_ok, "premise: the DCReg path must actually run"
        assert deg.is_degenerate, "premise: something must be flagged"
        assert deg.degenerate_mask[0] and deg.degenerate_mask[1], "x and y flagged"
        assert deg.cond_schur_t > 1e4

    def test_pcg_reproduces_the_plain_gauss_newton_result(self, near_degenerate_pair):
        """
        THE HEADLINE HONEST RESULT.  On geometry ill conditioned enough that
        cond(S_t) ~ 3.3e5 and three axes are flagged, running the paper's
        preconditioned CG changes the converged transform by 7.3e-6 — nothing,
        against a 0.6 m correction.  It solves the same normal equations to the
        same fixed point.  mode='pcg' buys conditioning, not mitigation.
        """
        target, source = near_degenerate_pair

        T_default = _engine(target).align(source)
        T_pcg = _engine(target).align(source, dcreg_mode='pcg')

        assert np.all(np.isfinite(T_pcg))
        assert np.allclose(T_default, T_pcg, atol=1e-4), (
            f"max diff {np.max(np.abs(T_default - T_pcg))}"
        )
        # Both recover the true offset; 'pcg' suppresses nothing.
        np.testing.assert_allclose(T_pcg[:3, 3], [-0.6, -0.2, -0.5], atol=1e-4)

    def test_clamped_suppresses_a_recoverable_correction(self, near_degenerate_pair):
        """
        The other honest result, and the reason kappa_threshold is a real
        tuning decision rather than a free win.

        The curvature of this surface makes x and y genuinely recoverable —
        plain Gauss-Newton nails the 0.6 m / 0.2 m offset to six figures.  But
        cond(S_t) ~ 3.3e5 puts both axes over a kappa_threshold of 10, so
        'clamped' regularizes them and the correction never happens: x ends at
        -0.0012 instead of -0.6.

        This is the method working as specified, not a bug.  It is recorded as a
        test because the failure mode of 'clamped' is silent — it returns a
        confident, finite, wrong transform — and anyone tuning kappa_threshold
        needs to know that is what is on the other side of the dial.
        """
        target, source = near_degenerate_pair

        T_default = _engine(target).align(source)
        T_clamped = _engine(target).align(source, dcreg_mode='clamped')

        assert np.all(np.isfinite(T_clamped))

        # Gauss-Newton recovers the lateral offset.
        np.testing.assert_allclose(T_default[:3, 3], [-0.6, -0.2, -0.5], atol=1e-4)

        # 'clamped' declines to, along exactly the flagged axes.
        assert abs(T_clamped[0, 3]) < 0.01, f"x = {T_clamped[0, 3]}"
        assert abs(T_clamped[1, 3]) < 0.01, f"y = {T_clamped[1, 3]}"
        # z is unflagged, so it is corrected normally.
        assert T_clamped[2, 3] == pytest.approx(-0.5, abs=1e-2)

        assert np.max(np.abs(T_default - T_clamped)) > 0.1, (
            "clamped should differ substantially from Gauss-Newton here"
        )


# --------------------------------------------------------------------------
# 10 - Default-path purity
# --------------------------------------------------------------------------

# Captured from align() on the well_conditioned_pair fixture BEFORE the dcreg
# wiring was written (git 8ba2d99, the D1 commit), at full float64 precision.
# np.array_equal against this pins that adding dcreg_mode executed no new code
# on the default path — not merely that the result stayed close.
PRE_CHANGE_ALIGN_REFERENCE = np.array([
    [0.9983036324712954, 0.04967032970764143, 0.03048017833242481, -0.1472187184465728],
    [-0.050266712643994446, 0.9985520106849926, 0.019238934321333206, 0.10585075667986074],
    [-0.029481895626922033, -0.0207362783741447, 0.999353963878726, -0.0776004122395889],
    [0.0, 0.0, 0.0, 1.0],
])


def test_default_path_is_bit_identical_to_pre_change_reference(well_conditioned_pair):
    target, source, _ = well_conditioned_pair

    T = _engine(target).align(source)

    assert np.array_equal(T, PRE_CHANGE_ALIGN_REFERENCE), (
        "the default align() path changed; max diff "
        f"{np.max(np.abs(T - PRE_CHANGE_ALIGN_REFERENCE))}"
    )


def test_existing_options_still_bit_identical_with_dcreg_absent(well_conditioned_pair):
    """The other two option paths must be untouched as well."""
    target, source, _ = well_conditioned_pair

    T_none = _engine(target).align(source, dcreg_mode=None)

    assert np.array_equal(T_none, PRE_CHANGE_ALIGN_REFERENCE)

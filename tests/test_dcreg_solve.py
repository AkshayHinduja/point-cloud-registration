"""
Unit tests for ``point_cloud_registration.degeneracy.dcreg_solve`` — the
DCReg-style targeted mitigation solve.

Clean-room implementation of the mitigation mathematics published in

    Hu et al., "DCReg: Decoupled Characterization for Efficient Degenerate
    LiDAR Registration", IJRR 2026, arXiv:2509.06285.

WHAT THESE TESTS ARE REALLY PINNING
-----------------------------------
The two modes are not two flavours of the same thing, and the tests are written
to keep that distinction visible:

* ``mode='pcg'`` is the paper's published solver.  It is a *preconditioned*
  Conjugate Gradient on the unmodified system ``H dx = -g``.  Preconditioning
  changes the path, never the fixed point: at full convergence it returns the
  plain Gauss-Newton step.  Tests 1 and 5 exist to pin exactly that, so nobody
  later mistakes this mode for regularization.  On a 6x6 system the Krylov space
  is complete after 6 iterations, so "full convergence" is the normal case and
  the truncation-based regularization story is close to vacuous here — see
  :class:`TestEarlyStop` for the measured numbers.

* ``mode='clamped'`` genuinely changes the minimum.  It adds the PSD spectral
  deficit ``A diag(clamped - aligned) A^T`` to the flagged aligned directions
  and to nothing else.  Test 2 pins that targeting property.
"""
import numpy as np
import pytest

from point_cloud_registration.degeneracy import (
    analyse_hessian_decoupled,
    dcreg_solve,
    _dcreg_preconditioner,
)

# --------------------------------------------------------------------------
# Fixtures — reuse the D1 golden-vector Hessian construction
# --------------------------------------------------------------------------

# Reference -> PCR DOF permutation: [roll,pitch,yaw,x,y,z] -> [tx,ty,tz,wx,wy,wz]
P_REF_TO_PCR = [3, 4, 5, 0, 1, 2]


def _golden_h_pcr() -> np.ndarray:
    """
    The ``dcreg_minimal_example`` synthetic weak-axis Hessian in PCR DOF order.

    Provenance and the transcription of the reference constants are documented
    in ``tests/test_degeneracy_decoupled.py``; this is the same construction.
    Under ``kappa_threshold=10`` it flags PCR indices 2 (tz) and 5 (wz / yaw),
    and its full 6x6 condition number is ~1122.

    LICENCE / PROVENANCE NOTE
    ------------------------
    The numbers below (the ``basis`` off-diagonals and the ``stiffness``
    vector) are the *inputs* of the DCReg project's published minimal example,
    reproduced here as published facts for one purpose only: to give this
    clean-room implementation an oracle to cross-check against the reference
    output printed in that project's README (Hu et al., "DCReg", IJRR 2026,
    arXiv:2509.06285).  They are test-fixture constants describing a synthetic
    problem, not an implementation copy — no reference algorithm source was
    read, and none of the solver logic in ``point_cloud_registration`` derives
    from anything but the published mathematics.
    """
    basis = np.eye(6)
    basis[0, 2] = 0.35
    basis[1, 5] = 0.15
    basis[2, 5] = 0.45
    basis[3, 2] = 0.20
    basis[4, 5] = 0.25

    stiffness = np.array([6.0, 4.5, 1.1, 5.0, 3.5, 0.18])

    J = basis @ np.diag(stiffness)
    H_ref = J.T @ J
    return H_ref[np.ix_(P_REF_TO_PCR, P_REF_TO_PCR)]


def _well_conditioned_h() -> np.ndarray:
    """Seeded SPD Hessian with no flagged axis under kappa_threshold=10."""
    rng = np.random.default_rng(17)
    J = rng.standard_normal((200, 6))
    return J.T @ J


def _seeded_g(seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(6)


def _gn_step(H: np.ndarray, g: np.ndarray) -> np.ndarray:
    """The plain Gauss-Newton step of the symmetrized system."""
    A = 0.5 * (np.asarray(H, float) + np.asarray(H, float).T)
    return np.linalg.solve(A, -np.asarray(g, float))


def _project_on_aligned_axes(deg, vec: np.ndarray) -> np.ndarray:
    """
    Decompose a 6-vector onto the per-block aligned eigenbases, returning a
    6-vector of coefficients in DOF order (so it lines up with
    ``deg.degenerate_mask``).
    """
    return np.concatenate([
        deg.aligned_basis_t.T @ vec[:3],
        deg.aligned_basis_R.T @ vec[3:],
    ])


# --------------------------------------------------------------------------
# 1 - PCG returns the Gauss-Newton step at full convergence
# --------------------------------------------------------------------------

class TestPcgIsGaussNewtonAtConvergence:
    """
    The honest-docstring property, pinned as a test.

    The paper's PCG solves the *unmodified* normal equations.  The
    preconditioner bounds the effective condition number and so controls how
    fast the iteration gets there; it does not move the minimum.  If a future
    change made mode='pcg' return something other than the GN step at
    convergence, that would be a behavioural regression, not an improvement.
    """

    def test_well_conditioned_matches_gauss_newton(self):
        H = _well_conditioned_h()
        g = _seeded_g()
        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0)
        assert deg.factorization_ok
        assert not deg.is_degenerate, "fixture should have nothing flagged"

        dx, info = dcreg_solve(H, g, deg, mode='pcg')

        np.testing.assert_allclose(dx, _gn_step(H, g), atol=1e-10)
        assert info['mode'] == 'pcg'
        assert info['pcg_converged'] is True
        assert info['used_fallback'] is False
        assert info['pcg_iterations'] is not None and info['pcg_iterations'] >= 1

    def test_flagged_axes_tight_tolerance_still_gauss_newton(self):
        """
        Even when axes ARE flagged — so the preconditioner really does clamp a
        near-zero curvature — a tight tolerance and a generous iteration budget
        land on the plain GN step.  Flagging changes the preconditioner, and the
        preconditioner does not change the answer.
        """
        H = _golden_h_pcr()
        g = _seeded_g()
        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0, kappa_target=10.0)
        assert deg.factorization_ok
        assert deg.degenerate_mask[2] and deg.degenerate_mask[5], "tz and yaw flagged"

        dx, info = dcreg_solve(
            H, g, deg, mode='pcg', pcg_tolerance=1e-14, pcg_max_iterations=50
        )

        np.testing.assert_allclose(dx, _gn_step(H, g), atol=1e-10)
        # Measured: converges in 6 iterations (Krylov space of a 6x6 system is
        # complete after 6 steps) to a relative residual of ~6e-17.
        assert info['used_fallback'] is False
        assert info['pcg_converged'] is True
        assert info['pcg_iterations'] <= 6

    def test_zero_gradient_gives_zero_step(self):
        H = _well_conditioned_h()
        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0)
        dx, info = dcreg_solve(H, np.zeros(6), deg, mode='pcg')
        np.testing.assert_allclose(dx, np.zeros(6), atol=0.0)
        assert info['pcg_converged'] is True


# --------------------------------------------------------------------------
# 2 - 'clamped' regularizes only the flagged directions
# --------------------------------------------------------------------------

class TestClampedIsTargeted:
    """
    ``mode='clamped'`` adds ``Gamma = A diag(clamped - aligned) A^T`` per block.
    ``clamped - aligned`` is zero on every unflagged axis by construction, so
    the regularization is *targeted*: it is Tikhonov-like along the flagged
    aligned axes and exactly zero elsewhere.
    """

    def setup_method(self):
        self.H = _golden_h_pcr()
        self.g = _seeded_g()
        self.deg = analyse_hessian_decoupled(
            self.H, kappa_threshold=10.0, kappa_target=10.0
        )
        self.dx_gn = _gn_step(self.H, self.g)
        self.dx_clamped, self.info = dcreg_solve(
            self.H, self.g, self.deg, mode='clamped'
        )

    def test_info_shape(self):
        assert self.info['mode'] == 'clamped'
        assert self.info['used_fallback'] is False
        assert self.info['pcg_iterations'] is None
        assert self.info['pcg_converged'] is None
        assert self.info['relative_residual'] is None

    def test_change_is_concentrated_on_flagged_directions(self):
        """
        Decompose ``dx_clamped - dx_gn`` on the aligned eigenbases.  The
        coefficients on the UNflagged axes must be small next to those on the
        flagged ones.

        Note this is not exactly zero and cannot be: the regularized system is
        still coupled, so ``dx_c - dx_gn = -H_reg^-1 Gamma dx_gn`` spreads the
        flagged-direction correction over the other axes through ``H_reg^-1``.
        Measured on this fixture: max unflagged coefficient 6.9e-2 against a
        minimum flagged coefficient of 1.23 — a ratio of 0.056.
        """
        mask = self.deg.degenerate_mask
        proj = _project_on_aligned_axes(self.deg, self.dx_clamped - self.dx_gn)

        max_unflagged = np.abs(proj[~mask]).max()
        min_flagged = np.abs(proj[mask]).min()

        assert max_unflagged / min_flagged < 0.15, (
            f"regularization is not targeted: max unflagged coefficient "
            f"{max_unflagged:.4g} vs min flagged {min_flagged:.4g}"
        )

    def test_step_along_flagged_axes_is_shorter_than_gauss_newton(self):
        """Clamping raises curvature, so the step along those axes shrinks."""
        mask = self.deg.degenerate_mask
        proj_c = _project_on_aligned_axes(self.deg, self.dx_clamped)
        proj_gn = _project_on_aligned_axes(self.deg, self.dx_gn)

        for i in np.flatnonzero(mask):
            assert abs(proj_c[i]) < abs(proj_gn[i]), (
                f"axis {i}: clamped step {proj_c[i]:.6g} is not shorter than "
                f"GN step {proj_gn[i]:.6g}"
            )
        # Measured: the tz coefficient drops 14.6 -> 0.19, yaw 1.30 -> 0.074.
        assert abs(proj_c[2]) < 0.1 * abs(proj_gn[2])

    def test_regularizer_is_psd_and_zero_on_unflagged_axes(self):
        """clamped_lambda - aligned_lambda is >= 0 and zero where unflagged."""
        deficit_t = self.deg.clamped_lambda_t - self.deg.aligned_lambda_t
        deficit_R = self.deg.clamped_lambda_R - self.deg.aligned_lambda_R
        assert (deficit_t >= 0.0).all()
        assert (deficit_R >= 0.0).all()
        np.testing.assert_allclose(deficit_t[~self.deg.degenerate_mask[:3]], 0.0, atol=0.0)
        np.testing.assert_allclose(deficit_R[~self.deg.degenerate_mask[3:]], 0.0, atol=0.0)

    @pytest.mark.parametrize("kappa_target", [1.0, 5.0, 10.0, 100.0, 1e4])
    def test_deficit_is_non_negative_for_any_kappa_target(self, kappa_target):
        """
        Clamping is MONOTONE, so the spectral deficit is a valid regularizer for
        every positive kappa_target — including kappa_target > kappa_threshold,
        where the target level ``lambda_max / kappa_target`` falls below the
        eigenvalue of an axis whose ratio sits between the two thresholds.  Such
        an axis keeps its own curvature (deficit exactly 0) instead of being
        lowered.
        """
        H = _golden_h_pcr()
        deg = analyse_hessian_decoupled(
            H, kappa_threshold=10.0, kappa_target=kappa_target
        )

        deficit_t = deg.clamped_lambda_t - deg.aligned_lambda_t
        deficit_R = deg.clamped_lambda_R - deg.aligned_lambda_R

        assert (deficit_t >= 0.0).all(), f"translation deficit negative: {deficit_t}"
        assert (deficit_R >= 0.0).all(), f"rotation deficit negative: {deficit_R}"
        assert (deg.clamped_lambda_t >= deg.aligned_lambda_t).all()
        assert (deg.clamped_lambda_R >= deg.aligned_lambda_R).all()

    @pytest.mark.parametrize("kappa_target", [1.0, 5.0, 10.0, 100.0, 1e4])
    def test_gamma_is_psd_for_any_kappa_target(self, kappa_target):
        """The regularizer dcreg_solve builds is positive semi-definite."""
        H = _golden_h_pcr()
        deg = analyse_hessian_decoupled(
            H, kappa_threshold=10.0, kappa_target=kappa_target
        )

        for basis, clamped, aligned in (
            (deg.aligned_basis_t, deg.clamped_lambda_t, deg.aligned_lambda_t),
            (deg.aligned_basis_R, deg.clamped_lambda_R, deg.aligned_lambda_R),
        ):
            Gamma = basis @ np.diag(clamped - aligned) @ basis.T
            eig = np.linalg.eigvalsh(0.5 * (Gamma + Gamma.T))
            assert (eig >= -1e-12).all(), f"Gamma is indefinite: {eig}"

    @pytest.mark.parametrize("kappa_target", [1.0, 5.0, 10.0, 100.0, 1e4])
    def test_clamped_solve_needs_no_fallback_on_a_pd_hessian(self, kappa_target):
        """
        PD input + PSD regularizer => PD regularized system, so np.linalg.solve
        succeeds and used_fallback stays False whatever kappa_target is.
        """
        H = _golden_h_pcr()
        assert (np.linalg.eigvalsh(H) > 0.0).all(), "fixture must be PD"

        deg = analyse_hessian_decoupled(
            H, kappa_threshold=10.0, kappa_target=kappa_target
        )
        dx, info = dcreg_solve(H, _seeded_g(), deg, mode='clamped')

        assert info['used_fallback'] is False
        assert np.all(np.isfinite(dx))

    def test_gentle_kappa_target_regularizes_less_than_the_default(self):
        """
        The behavioural meaning of kappa_target > kappa_threshold after the
        monotonicity fix: a gentler nudge, not an inverted one.  The step stays
        strictly closer to Gauss-Newton than the default clamp does.
        """
        H = _golden_h_pcr()
        g = _seeded_g()
        dx_gn = _gn_step(H, g)

        deg_default = analyse_hessian_decoupled(H, kappa_threshold=10.0)
        deg_gentle = analyse_hessian_decoupled(
            H, kappa_threshold=10.0, kappa_target=100.0
        )

        dx_default = dcreg_solve(H, g, deg_default, mode='clamped')[0]
        dx_gentle = dcreg_solve(H, g, deg_gentle, mode='clamped')[0]

        assert np.linalg.norm(dx_gentle - dx_gn) < np.linalg.norm(dx_default - dx_gn)

    def test_no_flagged_axis_reduces_to_gauss_newton(self):
        """With nothing flagged the deficit is zero and 'clamped' IS the GN step."""
        H = _well_conditioned_h()
        g = _seeded_g()
        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0)
        assert not deg.is_degenerate

        dx, info = dcreg_solve(H, g, deg, mode='clamped')

        np.testing.assert_allclose(dx, _gn_step(H, g), atol=1e-10)
        assert info['used_fallback'] is False


# --------------------------------------------------------------------------
# 3 - Preconditioner structure
# --------------------------------------------------------------------------

class TestPreconditionerShape:
    def test_block_diagonal_with_exactly_zero_coupling(self):
        """
        The preconditioner is assembled per block, so the two off-diagonal 3x3
        blocks are structurally zero — it deliberately does NOT approximate the
        coupling that the Schur complements factored out.
        """
        H = _golden_h_pcr()
        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0, kappa_target=10.0)
        P = _dcreg_preconditioner(deg)

        assert P.shape == (6, 6)
        np.testing.assert_array_equal(P[:3, 3:], np.zeros((3, 3)))
        np.testing.assert_array_equal(P[3:, :3], np.zeros((3, 3)))
        np.testing.assert_allclose(P, P.T, atol=1e-12)

    def test_identity_like_hessian_gives_inverse_curvature(self):
        """
        For a block-diagonal H with no flagged axis, the Schur complements are
        the diagonal blocks themselves and P is exactly their inverse.
        """
        H = np.diag([5.0, 5.0, 5.0, 4.0, 4.0, 4.0])
        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0)
        assert not deg.is_degenerate

        P = _dcreg_preconditioner(deg)

        np.testing.assert_allclose(P[:3, :3], np.eye(3) / 5.0, atol=1e-12)
        np.testing.assert_allclose(P[3:, 3:], np.eye(3) / 4.0, atol=1e-12)
        np.testing.assert_allclose(P @ H, np.eye(6), atol=1e-12)

    def test_uses_clamped_not_raw_spectrum(self):
        """
        On a flagged axis the preconditioner must use the CLAMPED eigenvalue —
        that is the whole point.  Using the raw one would give an enormous
        inverse curvature along the direction the data does not constrain.
        """
        H = _golden_h_pcr()
        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0, kappa_target=10.0)
        P = _dcreg_preconditioner(deg)

        expected_t = (
            deg.aligned_basis_t
            @ np.diag(1.0 / deg.clamped_lambda_t)
            @ deg.aligned_basis_t.T
        )
        np.testing.assert_allclose(P[:3, :3], expected_t, atol=1e-12)

        raw_t = (
            deg.aligned_basis_t
            @ np.diag(1.0 / deg.aligned_lambda_t)
            @ deg.aligned_basis_t.T
        )
        assert not np.allclose(P[:3, :3], raw_t, atol=1e-6), (
            "preconditioner appears to use the raw spectrum, not the clamped one"
        )

    @pytest.mark.parametrize("kappa_target", [10.0, 30.0, 100.0])
    def test_preconditioner_divides_block_condition_by_kappa_target(self, kappa_target):
        """
        The precise conditioning claim, and a guard against overstating it.

        The preconditioned block spectrum is ``lambda_i / clamped_i``: exactly 1
        on an unflagged axis, ``lambda_i * kappa_target / lambda_max`` on a
        flagged one.  So cond(B_t S_t) = cond(S_t) / kappa_target — the
        preconditioner DIVIDES the condition number by kappa_target rather than
        bounding it by kappa_target (or by kappa_target**2).  With
        cond(S_t) = 742.07 and kappa_target = 10 the result is 74.2, still ill
        conditioned.
        """
        H = _golden_h_pcr()
        deg = analyse_hessian_decoupled(
            H, kappa_threshold=10.0, kappa_target=kappa_target
        )
        P = _dcreg_preconditioner(deg)

        assert deg.cond_schur_t > 500.0, "fixture should be ill conditioned"

        eig = np.abs(np.linalg.eigvals(P[:3, :3] @ deg.S_t))
        cond_preconditioned = eig.max() / eig.min()

        assert cond_preconditioned == pytest.approx(
            deg.cond_schur_t / kappa_target, rel=1e-9
        )
        assert cond_preconditioned < deg.cond_schur_t

    def test_preconditioning_improves_but_does_not_fix_the_full_system(self):
        """Companion measurement on the full 6x6: cond 1122.3 -> 79.4."""
        H = _golden_h_pcr()
        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0, kappa_target=10.0)
        P = _dcreg_preconditioner(deg)
        A = 0.5 * (H + H.T)

        cond_raw = np.linalg.cond(A)
        eig = np.abs(np.linalg.eigvals(P @ A))
        cond_pre = eig.max() / eig.min()

        assert cond_raw == pytest.approx(1122.3, rel=1e-3)
        assert cond_pre == pytest.approx(79.4, rel=1e-2)
        assert cond_pre < cond_raw / 10.0


# --------------------------------------------------------------------------
# 4 - Fallbacks
# --------------------------------------------------------------------------

class TestFallbacks:
    def test_failed_factorization_uses_lstsq(self):
        """
        A Hessian whose diagonal blocks are singular has no Schur complement, so
        no DCReg mitigation is defined for it.  The solve degrades to a
        least-squares (minimum-norm) step rather than raising.
        """
        H = np.diag([5.0, 5.0, 5.0, 0.0, 0.0, 0.0])
        g = np.array([1.0, -2.0, 0.5, 0.0, 0.0, 0.0])
        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0)
        assert deg.factorization_ok is False

        for mode in ('pcg', 'clamped'):
            dx, info = dcreg_solve(H, g, deg, mode=mode)
            assert np.all(np.isfinite(dx)), f"{mode} produced non-finite dx"
            assert info['used_fallback'] is True
            assert info['mode'] == mode
            assert info['pcg_iterations'] is None
            # Minimum-norm solve: nothing moves along the unobserved block.
            np.testing.assert_allclose(dx[3:], np.zeros(3), atol=1e-12)

    def test_singular_full_hessian_with_b_in_range_stays_finite(self):
        """
        Rank-5 H whose diagonal blocks are still invertible, so the decoupled
        analysis succeeds and the solver really runs.  Whether PCG meets its
        tolerance or hands over to lstsq, the result must be finite.
        """
        rng = np.random.default_rng(23)
        Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        H = Q @ np.diag([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) @ Q.T
        H = 0.5 * (H + H.T)

        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0)
        assert deg.factorization_ok, "fixture: diagonal blocks must be invertible"

        v = rng.standard_normal(6)
        g = -(H @ v)  # b = -g = H v is in the range of H

        for mode in ('pcg', 'clamped'):
            dx, info = dcreg_solve(H, g, deg, mode=mode)
            assert np.all(np.isfinite(dx)), f"{mode} produced non-finite dx: {dx}"
            assert info['mode'] == mode

    def test_unconverged_pcg_reports_no_iterate_bookkeeping(self):
        """
        Contract: a PCG run that cannot meet its tolerance hands over to lstsq,
        and the PCG bookkeeping is cleared to show the iterate was not what got
        returned.  A tolerance of 1e-30 is unreachable in float64.
        """
        H = _golden_h_pcr()
        g = _seeded_g()
        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0)

        dx, info = dcreg_solve(
            H, g, deg, mode='pcg', pcg_tolerance=1e-30, pcg_max_iterations=10
        )

        assert info['used_fallback'] is True
        assert info['pcg_converged'] is None
        assert info['pcg_iterations'] is None
        assert info['relative_residual'] is None
        np.testing.assert_allclose(
            dx, np.linalg.lstsq(H, -g, rcond=None)[0], rtol=0, atol=1e-12
        )

    def test_failed_pcg_iterate_is_discarded_not_returned(self):
        """
        The teeth of the discard contract.

        Rank-5 H (diagonal blocks still invertible, so the analysis succeeds and
        the solver really runs) with the right-hand side deliberately pushed
        OUTSIDE range(H).  CG cannot converge, and its iterate blows up along
        the null direction: replaying the loop by hand gives a partial iterate of
        norm ~2.1e16.  What dcreg_solve returns is the lstsq solution, of norm
        ~2.48 — so the partial iterate was discarded rather than handed back.
        """
        rng = np.random.default_rng(23)
        Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        H = Q @ np.diag([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) @ Q.T
        H = 0.5 * (H + H.T)

        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0)
        assert deg.factorization_ok, "fixture: the solver must actually run"

        null_direction = Q[:, 0]                      # eigenvalue 0
        b = H @ rng.standard_normal(6) + 3.0 * null_direction
        g = -b

        dx, info = dcreg_solve(
            H, g, deg, mode='pcg', pcg_tolerance=1e-30, pcg_max_iterations=10
        )

        assert info['used_fallback'] is True
        assert info['pcg_iterations'] is None

        expected = np.linalg.lstsq(H, b, rcond=None)[0]
        np.testing.assert_allclose(dx, expected, rtol=0, atol=1e-12)

        # Had the partial iterate been returned, this would be ~2.1e16.
        assert np.linalg.norm(dx) < 10.0, (
            f"returned step has norm {np.linalg.norm(dx):.3e}; that looks like "
            "the discarded PCG iterate, not the lstsq fallback"
        )

    def test_fallback_result_solves_the_system_when_it_can(self):
        """The lstsq fallback is a real solve, not a zero-step surrender."""
        H = np.diag([5.0, 5.0, 5.0, 0.0, 0.0, 0.0])
        g = np.array([1.0, -2.0, 0.5, 0.0, 0.0, 0.0])
        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0)

        dx, _ = dcreg_solve(H, g, deg, mode='clamped')

        np.testing.assert_allclose(dx[:3], -g[:3] / 5.0, atol=1e-12)


# --------------------------------------------------------------------------
# 5 - Early stop
# --------------------------------------------------------------------------

class TestEarlyStop:
    """
    A loose tolerance stops the Krylov iteration before the space is complete,
    and the truncated iterate is what the method actually applies.

    HONEST MEASUREMENT, recorded so the claim is not oversold: on this fixture
    (cond_full ~1122) a tolerance of 1e-2 stops after 5 of the 6 iterations with
    a relative residual of 3.7e-6 — already four orders tighter than asked — and
    the returned step differs from the plain Gauss-Newton step by 2.4e-8 in
    relative norm.  Sweeping harsher fixtures (Schur condition numbers up to
    1.3e5) and looser tolerances (up to 1e-1) never produced a gap larger than
    ~1e-15: the iteration simply runs its 6 steps and lands on GN.  So at 6 DOF
    the "truncated-Krylov regularization" is a real mechanism but a numerically
    negligible one, and mode='pcg' should be understood as a *conditioning*
    improvement, not a mitigation.
    """

    def test_loose_tolerance_stops_before_the_krylov_space_is_complete(self):
        H = _golden_h_pcr()
        g = _seeded_g()
        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0, kappa_target=10.0)

        dx, info = dcreg_solve(
            H, g, deg, mode='pcg', pcg_tolerance=1e-2, pcg_max_iterations=10
        )

        assert info['pcg_converged'] is True
        assert info['used_fallback'] is False
        assert info['relative_residual'] <= 1e-2
        assert info['pcg_iterations'] < 6, (
            "expected the loose tolerance to stop the iteration early; got "
            f"{info['pcg_iterations']} iterations"
        )
        assert np.all(np.isfinite(dx))

    def test_truncated_iterate_is_still_essentially_the_gauss_newton_step(self):
        """The honest counterpart to the test above (see class docstring)."""
        H = _golden_h_pcr()
        g = _seeded_g()
        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0, kappa_target=10.0)

        dx_loose, _ = dcreg_solve(H, g, deg, mode='pcg', pcg_tolerance=1e-2)
        dx_gn = _gn_step(H, g)

        rel = np.linalg.norm(dx_loose - dx_gn) / np.linalg.norm(dx_gn)
        assert rel < 1e-6, (
            "documented behaviour changed: the truncated PCG iterate used to be "
            f"within 2.4e-8 of the GN step, now {rel:.3g}"
        )

    def test_relative_residual_reported_only_for_pcg_iterates(self):
        H = _golden_h_pcr()
        g = _seeded_g()
        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0)

        _, info = dcreg_solve(H, g, deg, mode='pcg')
        assert isinstance(info['relative_residual'], float)
        assert info['relative_residual'] >= 0.0


# --------------------------------------------------------------------------
# 6 - Mode validation
# --------------------------------------------------------------------------

class TestModeValidation:
    def test_unknown_mode_raises(self):
        H = _golden_h_pcr()
        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0)
        with pytest.raises(ValueError, match="mode"):
            dcreg_solve(H, _seeded_g(), deg, mode='foo')

    def test_unknown_mode_raises_even_when_factorization_failed(self):
        """Validation happens before the fallback, so a bad mode never passes."""
        H = np.diag([5.0, 5.0, 5.0, 0.0, 0.0, 0.0])
        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0)
        assert deg.factorization_ok is False
        with pytest.raises(ValueError, match="mode"):
            dcreg_solve(H, _seeded_g(), deg, mode='foo')

    @pytest.mark.parametrize(
        "bad_tolerance", [0.0, -1e-6, -1.0, float('nan'), float('inf')]
    )
    def test_non_positive_or_non_finite_tolerance_raises(self, bad_tolerance):
        H = _golden_h_pcr()
        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0)
        with pytest.raises(ValueError, match="pcg_tolerance"):
            dcreg_solve(H, _seeded_g(), deg, mode='pcg', pcg_tolerance=bad_tolerance)

    @pytest.mark.parametrize("bad_iterations", [10.5, -1, -10, 2.0, None, "10"])
    def test_bad_max_iterations_raises(self, bad_iterations):
        H = _golden_h_pcr()
        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0)
        with pytest.raises(ValueError, match="pcg_max_iterations"):
            dcreg_solve(
                H, _seeded_g(), deg, mode='pcg', pcg_max_iterations=bad_iterations
            )

    def test_pcg_parameters_validated_in_clamped_mode_too(self):
        """
        A nonsensical PCG configuration is rejected rather than silently ignored
        just because the chosen mode would not have used it.
        """
        H = _golden_h_pcr()
        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0)
        with pytest.raises(ValueError, match="pcg_tolerance"):
            dcreg_solve(H, _seeded_g(), deg, mode='clamped', pcg_tolerance=0.0)

    def test_zero_max_iterations_is_accepted(self):
        """
        Zero is a legal budget, not an error: the solver floors the loop at 6
        iterations regardless, which the docstring states.
        """
        H = _golden_h_pcr()
        deg = analyse_hessian_decoupled(H, kappa_threshold=10.0)
        dx, info = dcreg_solve(
            H, _seeded_g(), deg, mode='pcg', pcg_max_iterations=0
        )
        assert np.all(np.isfinite(dx))
        assert info['pcg_converged'] is True


# --------------------------------------------------------------------------
# Package export
# --------------------------------------------------------------------------

def test_exported_from_package_root():
    import point_cloud_registration as pcr

    assert hasattr(pcr, "dcreg_solve")

"""
A/B characterization of every degeneracy-mitigation strategy in ``align()``,
measured on shared fixtures.

This file exists to answer one question with executable evidence: *given the
same geometry, what does each mitigation strategy actually do to the answer?*
It is documentation, not regression cover — the unit behaviour of the solvers
lives in ``test_degeneracy.py``, ``test_dcreg_solve.py`` and
``test_degeneracy_decoupled.py``, and the wiring in ``test_registration_dcreg.py``
and ``test_registration_align.py``.  Every number in these docstrings was
measured from the code as committed, not predicted; where the measurement
contradicted the expectation it is the measurement that is written down.

Strategies under comparison
---------------------------
==== ==================================================== ===========================
key  configuration                                        what it is
==== ==================================================== ===========================
A    (nothing)                                            plain Gauss-Newton
B    ``lm_damping=True``                                  trace-scaled LM damping
C    ``use_solution_remapping=True, lm_damping=True``      Zhang & Singh SR, the
                                                          production configuration
D    ``dcreg_mode='pcg'``                                  DCReg preconditioned CG
E    ``dcreg_mode='clamped'``                              DCReg spectral clamping
==== ==================================================== ===========================

C pairs SR with damping because :func:`apply_sr_solve` inverts the Hessian it is
handed and several of these fixtures leave it singular; that pairing is what
``test_registration_align.py`` already documents as the usable SR configuration.

The headline
------------
On healthy geometry all five agree to 3e-7 and the choice is free.  On degenerate
geometry the strategies separate into three groups, and the separation is not the
one the labels suggest:

* **A** is the only one that can fail outright (``LinAlgError`` on an exactly
  singular H) and, wherever it runs at all, it is also the most accurate.
* **D (pcg)** tracks A to within 1e-5 on every fixture where A runs at all, and
  falls back to a minimum-norm ``lstsq`` step where it does not.  It solves the
  same normal equations; it buys conditioning, not mitigation.
* **B (lm)** converges to A's answer but *slowly* along weak directions — its
  shortfall is an iteration-budget artefact, not a change of fixed point
  (:func:`TestNearDegenerateOffset.test_lm_shortfall_is_budget_not_bias`).
* **C (sr)** and **E (clamped)** genuinely move the minimum.  Both suppress
  recoverable signal on the near-degenerate fixture; C additionally suppresses
  DOFs that are not weak at all, because the 6x6 criterion it is built on is not
  scale invariant (:class:`TestLeverArmDetection`).

Reproduce the whole comparison in one command::

    pytest tests/test_mitigation_ab.py -s -k summary_table

Determinism: every fixture is a fixed grid or a fixed-seed draw, every solve is
deterministic, and no test depends on wall-clock or thread count.
"""

import numpy as np
import pytest

from point_cloud_registration.plane_icp import PlaneICP
from point_cloud_registration.degeneracy import (
    analyse_hessian,
    analyse_hessian_decoupled,
)

from fixtures_degenerate import (
    build_flat_plane_pair,
    build_lever_arm_pair,
    build_near_degenerate_pair,
    build_well_conditioned_pair,
    LEVER_ARM_ROLL,
)


# Strategy key -> align() keyword arguments.  Ordered A..E as in the docstring.
STRATEGIES = {
    'A_gn': {},
    'B_lm': {'lm_damping': True},
    'C_sr_lm': {'use_solution_remapping': True, 'lm_damping': True},
    'D_pcg': {'dcreg_mode': 'pcg'},
    'E_clamped': {'dcreg_mode': 'clamped'},
}

# Strategies that survive an exactly rank-deficient Hessian (i.e. everything
# except plain Gauss-Newton).
MITIGATED = [k for k in STRATEGIES if k != 'A_gn']


# --------------------------------------------------------------------------
# Narrative-guarding bands
# --------------------------------------------------------------------------
# Each constant is the assertion that protects one headline claim made in a
# docstring below.  They live here, named, for two reasons:
#
#   * the dedicated test and the summary table (section 5) enforce literally
#     the same number, so the table's claim to catch a stale docstring is true
#     rather than aspirational; and
#   * a band is only useful if it is narrow enough to fail when the narrative
#     stops holding.  "SR essentially deletes the roll" is not guarded by a 5%
#     band, so these are set close to the measurements, not comfortably far
#     from them.  Widening one is a single visible line of diff.
#
# Every comment gives the value measured on 2026-07-31 on this machine
# (numpy/LAPACK as installed); the constant is the pin, the comment is the
# observation.
SR_ROLL_RECOVERY_MAX = 0.01            # measured 0.0011858
SR_VERTICAL_RECOVERY_MAX = 0.01        # measured 0.0045010
CLAMPED_LATERAL_RECOVERY_MAX = 0.05    # measured 0.0020720 (x), 0.0071380 (y)
LM_30_ITER_BAND = (0.15, 0.30)         # measured 0.2138623
LM_1000_ITER_MIN = 0.98                # measured 0.9946405
FULL_RECOVERY_MIN = 0.95               # measured 0.99718-1.00000


def _engine(target, max_iter=30):
    e = PlaneICP(max_iter=max_iter, max_dist=2.0, tol=1e-6)
    e.set_target(target)
    return e


# Several of these solves are re-run by half a dozen tests each (and again by
# the summary table).  Every (cloud pair, strategy, budget) triple is a pure
# deterministic function, so it is memoized on the raw cloud bytes; a copy is
# handed out so no test can perturb another's result.  This is a speed measure
# only — deleting the cache changes no assertion, just the runtime.
_ALIGN_CACHE = {}


def _align(target, source, key, max_iter=30):
    cache_key = (target.tobytes(), source.tobytes(), key, max_iter)
    if cache_key not in _ALIGN_CACHE:
        _ALIGN_CACHE[cache_key] = _engine(target, max_iter=max_iter).align(
            source, **STRATEGIES[key]
        )
    return _ALIGN_CACHE[cache_key].copy()


def _hessian(target, source):
    """The 6x6 Hessian at the identity — the matrix both analyses consume."""
    H = _engine(target).calc_H_g_e2(np.eye(4), source.astype(np.float32))[0]
    return np.asarray(H, dtype=float)


def _yaw(T):
    return float(np.arctan2(T[1, 0], T[0, 0]))


def _rotvec(T):
    """
    Axis-angle vector of T's rotation (SO(3) log), NumPy only.

    Used instead of scipy so this file has the same dependency footprint as the
    library it tests.
    """
    R = T[:3, :3]
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    axis_times_two_sin = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])
    if theta < 1e-12:
        return 0.5 * axis_times_two_sin
    return theta / (2.0 * np.sin(theta)) * axis_times_two_sin


def _recovered_fraction(T, true_translation):
    """
    Per-axis fraction of the known translation offset that ``T`` recovered.

    1.0 = fully recovered, 0.0 = the correction never happened.  This is the
    unit every comparison below is stated in, because "how much of the signal
    did the strategy keep?" is the question a threshold choice is really about.
    """
    return T[:3, 3] / np.asarray(true_translation, dtype=float)


def _dominant_dof(eigenvector):
    return int(np.argmax(np.abs(eigenvector)))


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def well_conditioned_pair():
    return build_well_conditioned_pair()


@pytest.fixture
def flat_plane_pair():
    return build_flat_plane_pair()


@pytest.fixture
def near_degenerate_pair():
    return build_near_degenerate_pair()


@pytest.fixture
def lever_arm_pair():
    return build_lever_arm_pair()


# ==========================================================================
# 1 - Well-conditioned geometry: nobody pays anything
# ==========================================================================

class TestWellConditionedParity:
    """
    When the problem is healthy, the mitigation strategies are free.

    MEANING: none of these options is a trade-off you have to think about on
    good data.  Every cost documented in the rest of this file is a cost paid
    *only* where the geometry is actually degenerate, so enabling a strategy
    cannot be judged from a well-conditioned benchmark — it will show nothing.
    """

    @pytest.mark.parametrize('key', MITIGATED)
    def test_every_strategy_matches_plain_gauss_newton(
        self, well_conditioned_pair, key
    ):
        """
        All five strategies converge to the same transform on the corner cloud.

        Measured max |T_strategy - T_gn|:
            B_lm 2.98e-07, C_sr_lm 2.98e-07, D_pcg 4.44e-09, E_clamped 0.0
            (E is bit-identical: nothing is flagged, so the added regularizer is
            the zero matrix and the solve is literally the Gauss-Newton one.)
        Asserted at atol 1e-3, ~3000x looser than the largest measured gap.
        """
        target, source, _ = well_conditioned_pair

        T_gn = _align(target, source, 'A_gn')
        T = _align(target, source, key)

        assert np.allclose(T_gn, T, atol=1e-3), (
            f"{key} diverged from plain GN; max diff {np.max(np.abs(T_gn - T))}"
        )

    def test_neither_analysis_flags_anything_here(self, well_conditioned_pair):
        """
        The premise of the parity test, and the reason the strategies agree.

        Measured: the 6x6 analysis reports condition number 11.37 with an
        all-False mask; the decoupled analysis reports cond(S_t) 7.75,
        cond(S_R) 1.16 and an all-False mask at kappa_threshold=10.
        """
        target, source, _ = well_conditioned_pair
        H = _hessian(target, source)

        deg6 = analyse_hessian(H)
        degd = analyse_hessian_decoupled(H, kappa_threshold=10.0)

        assert not deg6.is_degenerate, f"6x6 mask {deg6.degenerate_mask}"
        assert degd.factorization_ok
        assert not degd.is_degenerate, f"decoupled mask {degd.degenerate_mask}"
        assert degd.cond_schur_t < 10.0
        assert degd.cond_schur_R < 10.0


# ==========================================================================
# 2 - Flat plane: exact 3-DOF degeneracy
# ==========================================================================

class TestFlatPlaneExactDegeneracy:
    """
    Exact rank deficiency: tx, ty and wz are not weakly observed, they are
    *unobserved*.  Their Jacobian columns are identically zero.

    MEANING, and it is a narrower result than it looks: on this fixture the
    unobservable-direction drift is exactly 0.0 for **every** strategy, and none
    of them can take any credit for it.  A zero Jacobian column produces a zero
    gradient entry, so any solver that does not manufacture a step out of
    nothing leaves those DOFs alone — LM damping, SR projection, the lstsq
    fallback that both DCReg modes take here, all of them.  The only thing that
    separates the strategies on this fixture is whether they run at all.

    So this fixture proves *robustness* (four of five return a finite, sane
    transform where plain Gauss-Newton raises) and proves nothing whatsoever
    about *mitigation quality*.  For that, see
    :class:`TestNearDegenerateOffset`, where the weak directions carry real
    gradient and the strategies finally disagree.
    """

    def test_plain_gauss_newton_raises(self, flat_plane_pair):
        """
        Strategy A on an exactly singular H: ``np.linalg.solve`` refuses.

        Documented as behaviour, not as a defect — the caller is being told the
        problem is not solvable as posed, which is more useful than a
        confidently wrong answer.
        """
        target, source = flat_plane_pair
        with pytest.raises(np.linalg.LinAlgError):
            _align(target, source, 'A_gn')

    def test_hessian_is_exactly_rank_three(self, flat_plane_pair):
        """
        Premise: three eigenvalues are exactly zero, and the corresponding rows
        of H *and* entries of g are exactly zero too.

        Measured: eigenvalues [0, 0, 0, 121, 1210, 1210]; max |H[rows 0,1,5]| =
        0.0 and g[[0,1,5]] = [0, 0, 0] exactly, because the estimated plane
        normal comes out as exactly (0, 0, 1).  That exactness is why the drift
        numbers below are 0.0 rather than merely small.  Asserted at 1e-12
        rather than on equality so a LAPACK that returns a 1e-17 normal
        component cannot break the file over a difference that means nothing.
        """
        target, source = flat_plane_pair
        H, g, _ = _engine(target).calc_H_g_e2(
            np.eye(4), source.astype(np.float32)
        )
        H = np.asarray(H, dtype=float)
        g = np.asarray(g, dtype=float)

        # 1e-6 sits ~8 orders below the smallest *non*-zero eigenvalue (121)
        # and ~6 above eigh's round-off on a matrix of norm 1210.
        assert np.count_nonzero(np.linalg.eigvalsh(H) <= 1e-6) == 3
        assert np.max(np.abs(H[[0, 1, 5], :])) < 1e-12
        assert np.all(np.abs(g[[0, 1, 5]]) < 1e-12)

    @pytest.mark.parametrize('key', MITIGATED)
    def test_finite_with_zero_unobservable_drift(self, flat_plane_pair, key):
        """
        Per strategy: finite transform, no drift in x / y / yaw, z converged.

        Measured (all four strategies identical to print precision):
            in-plane |t_xy| = 0.0 exactly, yaw = 0.0 exactly
            z:  B_lm -0.49999994, C_sr_lm -0.49999994,
                D_pcg -0.5 exactly, E_clamped -0.5 exactly
        (B and C stop 6e-8 short because ``align()`` discards the final
        sub-tolerance step, and damping makes that step non-zero; D and E take
        the undamped lstsq step, which solves this quadratic exactly on the
        first iteration and then breaks on dx = 0.)

        Asserted at 1e-9 / 1e-3 rather than on equality so the test does not
        break on a future BLAS that rounds the last bit differently.
        """
        target, source = flat_plane_pair

        T = _align(target, source, key)

        assert np.all(np.isfinite(T)), f"{key} returned a non-finite transform"
        assert abs(T[0, 3]) < 1e-9, f"{key} x drift {T[0, 3]}"
        assert abs(T[1, 3]) < 1e-9, f"{key} y drift {T[1, 3]}"
        assert abs(_yaw(T)) < 1e-9, f"{key} yaw drift {_yaw(T)}"
        assert T[2, 3] == pytest.approx(-0.5, abs=1e-3), f"{key} z {T[2, 3]}"

    def test_the_two_dcreg_modes_never_reach_their_own_solvers(
        self, flat_plane_pair
    ):
        """
        Why D and E are indistinguishable here: neither Schur complement exists.

        H_tt is rank 1 and H_ww is rank 2, so ``factorization_ok`` is False and
        ``dcreg_solve`` takes its ``lstsq`` fallback on every iteration.  The
        clean numbers above are the minimum-norm property of
        ``numpy.linalg.lstsq``, not a DCReg mitigation, and the two modes return
        bit-identical transforms because they ran identical code.
        """
        target, source = flat_plane_pair
        H = _hessian(target, source)

        degd = analyse_hessian_decoupled(H, kappa_threshold=10.0)
        assert degd.factorization_ok is False

        np.testing.assert_array_equal(
            _align(target, source, 'D_pcg'),
            _align(target, source, 'E_clamped'),
        )


# ==========================================================================
# 3 - Near-degenerate but recoverable: the decision-relevant case
# ==========================================================================

# True correction for the near-degenerate fixture (source is displaced
# +0.6/+0.2/+0.5, so align() must undo it).
NEAR_DEGENERATE_TRUTH = np.array([-0.6, -0.2, -0.5])


class TestNearDegenerateOffset:
    """
    A gently curved surface, cond(S_t) ~ 3.3e5, carrying a *genuine* 0.6 m
    lateral offset that the data does support.

    MEANING: this is the fixture that prices every strategy, because here the
    weak directions are weak-but-real.  A strategy's recovered fraction is
    exactly the fraction of a true correction it is willing to make.  Plain
    Gauss-Newton recovers 100% of it; the question each strategy answers is how
    much of that it gives back in exchange for boundedness.

    Measured recovered fraction (1.0 = fully recovered), truth (-0.6, -0.2, -0.5):

        ========= ========== ========== ==========
        strategy  x          y          z
        ========= ========== ========== ==========
        A_gn       1.000004   1.000022   1.000003
        B_lm       0.2139     0.2373     1.0044      (30 iterations)
        B_lm       0.9946     0.9999     1.0000      (1000 iterations)
        C_sr_lm   -0.000456  -0.000539   0.004501
        D_pcg      1.000010   1.000054   1.000003
        E_clamped  0.002072   0.007138   1.004983
        ========= ========== ========== ==========

    Read the z column first.  z is the *well observed* direction here — the
    decoupled analysis does not flag it (its aligned eigenvalue is the block
    maximum, ratio 1.0) — and four of five strategies recover it in full.
    C_sr_lm recovers 0.45% of it.  That is the single most consequential number
    in this file: the production configuration discarded a healthy DOF, not a
    degenerate one, because the Zhang & Singh criterion compares an eigenvalue
    against ``sqrt(lambda_max/lambda_min)`` = 1816.9 here and z's eigenvalue is
    419.7.  See :meth:`test_sr_also_suppresses_the_well_observed_axis`.
    """

    def test_fixture_carries_recoverable_lateral_signal(
        self, near_degenerate_pair
    ):
        """
        Premise: the offset really is recoverable, so suppressing it really is a
        loss.  Plain Gauss-Newton recovers all three components to ~6 figures.
        """
        target, source = near_degenerate_pair

        T = _align(target, source, 'A_gn')

        np.testing.assert_allclose(
            T[:3, 3], NEAR_DEGENERATE_TRUTH, atol=1e-3
        )

    def test_pcg_recovers_the_offset_like_plain_gauss_newton(
        self, near_degenerate_pair
    ):
        """
        D tracks A to 1e-5 on geometry with cond(S_t) ~ 3.3e5 and three flagged
        axes.

        MEANING: ``dcreg_mode='pcg'`` is not a mitigation you can rely on to
        bound anything.  Preconditioning changes the path of the iteration, not
        its fixed point, so on a 6x6 system that always runs to convergence it
        returns the Gauss-Newton step.  Choose it when you want the *analysis*
        and better-conditioned arithmetic; do not choose it expecting the
        estimate to be protected.

        Measured residual against the truth: 6e-6 / 1.1e-5 / 1e-6 m.
        """
        target, source = near_degenerate_pair

        T = _align(target, source, 'D_pcg')

        residual = np.abs(T[:3, 3] - NEAR_DEGENERATE_TRUTH)
        assert np.all(residual < 1e-3), f"residual {residual}"

    def test_clamped_suppresses_the_recoverable_lateral_offset(
        self, near_degenerate_pair
    ):
        """
        E at kappa_threshold=10 keeps ≈0.21% of the x offset and ≈0.71% of y
        (2026-07-31, this machine), while recovering z (unflagged) in full.

        MEANING: this is the recoverable-signal-versus-boundedness trade, and it
        is the honest cost of ``'clamped'``.  kappa_threshold is the dial: every
        axis it flags has its correction largely deleted, whether or not the data
        would have supported it.  The failure mode is silent — a finite,
        confident, wrong transform — so the threshold must be chosen against the
        geometry you expect, not left at a default because the default is safe.

        Asserted to < 5% (``CLAMPED_LATERAL_RECOVERY_MAX``), an order of
        magnitude wider than the measurements.  Unlike the SR bands, 5% is
        genuinely what the narrative needs here: the claim is "clamped
        suppresses the correction", and keeping 5% of a 0.6 m offset would
        still be suppression.  It is not tightened to the measurement because
        nothing in the story depends on the difference between 0.2% and 3%.
        """
        target, source = near_degenerate_pair

        frac = _recovered_fraction(
            _align(target, source, 'E_clamped'), NEAR_DEGENERATE_TRUTH
        )

        assert abs(frac[0]) < CLAMPED_LATERAL_RECOVERY_MAX, (
            f"x recovered fraction {frac[0]}"
        )
        assert abs(frac[1]) < CLAMPED_LATERAL_RECOVERY_MAX, (
            f"y recovered fraction {frac[1]}"
        )
        assert frac[2] == pytest.approx(1.0, abs=0.05), (
            f"z should be untouched, recovered fraction {frac[2]}"
        )

    def test_sr_also_suppresses_the_well_observed_axis(
        self, near_degenerate_pair
    ):
        """
        C suppresses x and y like E does — and z as well, which E does not.

        Measured recovered fractions ≈ x -0.046%, y -0.054%, **z 0.45%**
        (2026-07-31, this machine); z asserted to < 1%.  Every other strategy
        recovers z at 100.0-100.5%, asserted to within 5% of 1.0.

        MEANING: the 6x6 criterion flagged a direction that is not weak.  Its
        threshold is ``sqrt(lambda_max/lambda_min)`` = 1816.9, an absolute
        eigenvalue floor derived from a unitless ratio; the tz-dominated
        eigen-direction has eigenvalue 419.7 and so falls under it, even though
        within the translation block it *is* the maximum (the decoupled analysis
        gives it ratio 1.0 and does not flag it).  This is the scale-dependence
        that ``analyse_hessian``'s own docstring warns about, priced: on this
        fixture it costs a full 0.5 m vertical correction that the data
        supported.

        Not a bug in ``apply_sr_solve`` — it faithfully zeroes what it is told
        to zero.  It is a property of the detection criterion feeding it, and it
        is the strongest argument in this file for using
        ``analyse_hessian_decoupled`` (or a calibrated
        ``sr_lambda_threshold``) rather than the adaptive 6x6 default.
        """
        target, source = near_degenerate_pair
        H = _hessian(target, source)

        deg6 = analyse_hessian(H)
        degd = analyse_hessian_decoupled(H, kappa_threshold=10.0)

        # The 6x6 analysis flags a tz-dominated direction ...
        flagged_tz = [
            i for i in range(6)
            if deg6.degenerate_mask[i] and _dominant_dof(deg6.eigenvectors[:, i]) == 2
        ]
        assert flagged_tz, (
            f"expected a flagged tz direction; mask {deg6.degenerate_mask}, "
            f"eigenvalues {deg6.eigenvalues}"
        )
        # ... whose eigenvalue is large in absolute terms; it is flagged only
        # because the adaptive threshold is larger still.
        assert deg6.eigenvalues[flagged_tz[0]] > 100.0
        assert deg6.lambda_threshold > deg6.eigenvalues[flagged_tz[0]]

        # ... while the decoupled analysis calls tz the best-observed
        # translation axis and leaves it alone.
        assert degd.factorization_ok
        assert not degd.degenerate_mask[2], (
            f"decoupled translation mask {degd.degenerate_mask[:3]}"
        )

        # And the consequence for the estimate.
        frac_sr = _recovered_fraction(
            _align(target, source, 'C_sr_lm'), NEAR_DEGENERATE_TRUTH
        )
        frac_clamped = _recovered_fraction(
            _align(target, source, 'E_clamped'), NEAR_DEGENERATE_TRUTH
        )
        assert abs(frac_sr[2]) < SR_VERTICAL_RECOVERY_MAX, (
            f"SR kept {frac_sr[2]:.3%} of the vertical correction; the claim "
            f"this test documents is that it keeps essentially none"
        )
        assert frac_clamped[2] == pytest.approx(1.0, abs=0.05), (
            "the contrast only means something if 'clamped' does recover z; "
            f"got {frac_clamped[2]}"
        )

    def test_lm_shortfall_is_budget_not_bias(self, near_degenerate_pair):
        """
        B looks like a suppressor at max_iter=30 and is not one.

        Measured recovered fraction in x, against iteration budget
        (2026-07-31, this machine):
            30 -> 0.2139, 100 -> 0.4977, 300 -> 0.8121, 1000 -> 0.9946,
            3000 -> 0.9997
        Asserted: 30 iterations inside (0.15, 0.30), 1000 iterations > 0.98.
        Both bands guard the narrative rather than merely the arithmetic — the
        claim is "clearly incomplete at 30, essentially complete at 1000", and
        a run that landed at 0.05 or at 0.9 after 30 iterations would mean
        something different and should fail.

        MEANING: LM's fixed point is still the true minimum — damping shrinks
        each step but does not move where the steps stop.  The trace-scaled
        lambda here is ~0.14 against a weak-direction eigenvalue of ~1.3e-3, so
        each iteration closes only ~1% of the remaining lateral error and 30
        iterations are nowhere near enough.  This is a materially different
        failure from C and E: LM under-converges *visibly* (it is still moving
        when the budget runs out), whereas SR and clamped converge cleanly to a
        different answer.  Raising max_iter fixes LM; nothing fixes a threshold
        that was set too tight.

        The 1000-iteration run is the slowest thing in this file (0.3-1.1 s
        depending on BLAS threading), which is why the budget stops there
        rather than at the 3000 iterations needed for the last 0.05%.
        """
        target, source = near_degenerate_pair

        frac_30 = _recovered_fraction(
            _align(target, source, 'B_lm', max_iter=30), NEAR_DEGENERATE_TRUTH
        )
        frac_1000 = _recovered_fraction(
            _align(target, source, 'B_lm', max_iter=1000), NEAR_DEGENERATE_TRUTH
        )

        lo, hi = LM_30_ITER_BAND
        assert lo < frac_30[0] < hi, (
            f"30-iteration fraction {frac_30[0]}, expected in ({lo}, {hi})"
        )
        assert frac_1000[0] > LM_1000_ITER_MIN, (
            f"1000-iteration fraction {frac_1000[0]}"
        )
        assert frac_1000[0] > frac_30[0]


# ==========================================================================
# 4 - Detection A/B under a lever arm
# ==========================================================================

class TestLeverArmDetection:
    """
    ``analyse_hessian`` (6x6) versus ``analyse_hessian_decoupled`` on a real
    cloud pair placed 50 m from the origin.

    This is a genuine point-cloud pair, not an H-level synthetic: the corner
    fixture rigidly translated +50 m in x, with the source rolled 0.01 rad about
    the cloud's own centroid.  The *local* geometry — and therefore the local
    observability — is identical to the well-conditioned fixture; the only thing
    that changed is the lever arm in ``p x n``, which inflates the wy and wz
    Hessian diagonals by 246x and 279x and leaves wx alone (the offset is
    parallel to x, so it adds no moment arm about x).

    MEANING: the 6x6 test flags roll as degenerate here and roll is fully
    recoverable.  The decoupled test does not flag it.  This is the concrete
    cost of comparing an eigenvalue against a unitless threshold when the
    eigenvalues have been inflated by a squared length — and it is not
    hypothetical, because the production configuration (C) is driven by that
    classification and throws the whole roll correction away.

    Honesty about what the 6x6 does get right: it is not blind here.  Two of the
    directions it flags are genuinely ambiguous — the ty/tz-dominated
    eigen-directions with eigenvalues 0.325 and 0.363, because translating a
    cloud 50 m away is nearly the same observation as rotating it about the
    origin — and the decoupled analysis flags those two as well.  The defect is
    over-flagging: 4 of 6 flagged, and the extra two are the *best*-observed
    axis of each block (tx and wx, Schur ratio 1.00 and 1.18 respectively).
    Only wx is demonstrated recoverable below, because this fixture carries no
    tx offset to recover; the tx claim rests on its Schur ratio alone.
    """

    def test_the_lever_arm_is_what_creates_the_disagreement(self):
        """
        Control: the same cloud, the same roll, at the origin instead of 50 m out.

        Measured at offset 0: both analyses report an all-False mask, cond_full
        126.8.  At offset 50 m: cond_full 3.52e6, and the 6x6 mask becomes
        [tx ty tz wx].  The rotation block's own diagonal shows exactly where
        that came from — H_ww diag goes from [4296, 4488, 3903] to
        [4295, 1.10e6, 1.09e6]: wy gains 246x and wz 279x, while wx gains
        nothing at all (it *drops* 0.02%).

        MEANING: nothing about the *geometry* got harder.  The disagreement is
        manufactured entirely by where the cloud sits relative to the origin of
        the parameterization, which is precisely the quantity a degeneracy test
        should be invariant to.
        """
        target0, source0 = build_lever_arm_pair(offset=0.0)
        target50, source50 = build_lever_arm_pair(offset=50.0)

        H0 = _hessian(target0, source0)
        H50 = _hessian(target50, source50)

        assert not analyse_hessian(H0).is_degenerate
        degd0 = analyse_hessian_decoupled(H0, kappa_threshold=10.0)
        assert not degd0.is_degenerate

        # The rotation Schur complement is essentially invariant to the move:
        # measured cond(S_R) 1.1781 at the origin, 1.1774 at 50 m — a 0.06%
        # change, against a 3.5e6/126.8 = 28000x change in cond(H).  That is
        # scale invariance doing exactly what it is advertised to do.
        degd50 = analyse_hessian_decoupled(H50, kappa_threshold=10.0)
        assert degd50.cond_schur_R == pytest.approx(degd0.cond_schur_R, rel=0.01)
        assert degd50.cond_full / degd0.cond_full > 1000.0

        # wx picks up no lever arm; wy gains 246x and wz 279x.
        assert H50[3, 3] == pytest.approx(H0[3, 3], rel=0.01)
        assert H50[4, 4] / H0[4, 4] > 100.0
        assert H50[5, 5] / H0[5, 5] > 100.0

        assert analyse_hessian(H50).is_degenerate, (
            "premise: the 6x6 analysis changes its mind once the cloud moves"
        )

    def test_the_two_analyses_disagree_about_roll(self, lever_arm_pair):
        """
        6x6: roll flagged.  Decoupled: roll clean.

        Measured 6x6 spectrum [0.325, 0.363, 390.7, 892.1, 1.051e6, 1.145e6]
        against an adaptive threshold of 1876.7 — so the wx-dominated direction
        (eigenvalue 892.1, |v_wx| = 0.998) falls under it and is flagged.

        Measured decoupled rotation block: aligned eigenvalues
        [851.4, 1002.4, 913.6] for [wx, wy, wz], cond(S_R) = 1.18, nothing
        flagged.  The Schur complement removed the lever-arm inflation, because
        the part of a wy/wz rotation that a 50 m-distant cloud sees is mostly a
        translation and translation already explains it.  What is left is the
        rotational information the rotation block genuinely owns, and that is
        near-uniform across the three axes.
        """
        target, source = lever_arm_pair
        H = _hessian(target, source)

        deg6 = analyse_hessian(H)
        degd = analyse_hessian_decoupled(H, kappa_threshold=10.0)

        flagged_wx = [
            i for i in range(6)
            if deg6.degenerate_mask[i] and abs(deg6.eigenvectors[3, i]) > 0.99
        ]
        assert flagged_wx, (
            "expected the 6x6 analysis to flag a roll-dominated direction; "
            f"mask {deg6.degenerate_mask}, eigenvalues {deg6.eigenvalues}"
        )

        assert degd.factorization_ok
        assert not degd.degenerate_mask[3], (
            f"decoupled rotation mask {degd.degenerate_mask[3:]}"
        )
        assert degd.cond_schur_R < 2.0, f"cond(S_R) = {degd.cond_schur_R}"
        # And the disagreement is confined to the rotation block: both analyses
        # flag translation directions.
        assert degd.degenerate_mask[1] and degd.degenerate_mask[2]

    def test_roll_is_recoverable_so_the_6x6_flag_is_a_false_positive(
        self, lever_arm_pair
    ):
        """
        Ground truth settles it: the fixture's roll is 0.01 rad and A recovers
        it exactly.

        Measured recovered fraction of the true roll (2026-07-31, this machine):
            A_gn 1.000000, D_pcg 0.999999, B_lm 0.997220, E_clamped 0.997180,
            **C_sr_lm 0.001186**.
        Asserted: the four recoverers to > 0.95, C_sr_lm to < 0.01.  The 1%
        band is the narrative — "SR essentially deletes the roll" — not a
        margin around the measurement; a run that kept even 5% of the
        correction would be a different story and should fail here.

        MEANING: solution remapping deleted a 0.01 rad correction that four
        other strategies made in full, because its detector mistook a missing
        lever arm for missing information.  If you run SR on data where the
        registration frame origin is far from the cloud — a georeferenced frame,
        a sensor-to-body offset, any map-frame ICP — expect this, and either
        supply a calibrated ``sr_lambda_threshold`` or use the decoupled
        analysis instead.
        """
        target, source = lever_arm_pair

        fractions = {
            key: _rotvec(_align(target, source, key))[0] / -LEVER_ARM_ROLL
            for key in STRATEGIES
        }

        for key in ('A_gn', 'D_pcg', 'B_lm', 'E_clamped'):
            assert fractions[key] > FULL_RECOVERY_MIN, (
                f"{key} recovered only {fractions[key]:.4f} of the roll"
            )
        assert abs(fractions['C_sr_lm']) < SR_ROLL_RECOVERY_MAX, (
            f"C_sr_lm recovered {fractions['C_sr_lm']:.4f} of the roll; the "
            f"claim this test documents is that SR essentially deletes it"
        )


# ==========================================================================
# 5 - Informational: the whole comparison in one command
# ==========================================================================

# MOSTLY INFORMATIONAL TEST.  Its purpose is to print the strategy x fixture
# matrix so a human can re-derive every number quoted in the docstrings above
# with:
#
#     pytest tests/test_mitigation_ab.py -s -k summary_table
#
# Beyond finiteness it re-asserts the three headline claims of this file
# against the *same named bands* the dedicated tests use (SR_ROLL_RECOVERY_MAX,
# SR_VERTICAL_RECOVERY_MAX, LM_30_ITER_BAND), recomputed here from the printed
# table rather than from a separate solve.  That is what makes "if a number
# here disagrees with a docstring, the docstring is stale" a true statement
# instead of an aspiration: the printed table and the assertions cannot drift
# apart, because they are the same numbers.
def test_summary_table(capsys):
    """
    Print (finite?, unobservable drift, recovered fraction) per strategy, and
    re-assert the file's three headline bands from the printed values.
    """
    rows = []
    headline = {}

    target, source, _ = build_well_conditioned_pair()
    T_gn_wc = _align(target, source, 'A_gn')
    for key in STRATEGIES:
        T = _align(target, source, key)
        rows.append((
            'well-conditioned', key, np.all(np.isfinite(T)),
            f"maxdiff_vs_gn={np.max(np.abs(T - T_gn_wc)):.2e}",
        ))

    target, source = build_flat_plane_pair()
    for key in STRATEGIES:
        try:
            T = _align(target, source, key)
        except np.linalg.LinAlgError as exc:
            rows.append(('flat-plane', key, False, f"raised LinAlgError: {exc}"))
            continue
        drift = float(np.hypot(T[0, 3], T[1, 3]))
        rows.append((
            'flat-plane', key, np.all(np.isfinite(T)),
            f"inplane_drift={drift:.3e} yaw={_yaw(T):+.3e} z={T[2, 3]:+.8f}",
        ))

    target, source = build_near_degenerate_pair()
    for key in STRATEGIES:
        T = _align(target, source, key)
        frac = _recovered_fraction(T, NEAR_DEGENERATE_TRUTH)
        headline[('near-degenerate', key)] = frac
        rows.append((
            'near-degenerate', key, np.all(np.isfinite(T)),
            f"recovered=[{frac[0]:+.6f} {frac[1]:+.6f} {frac[2]:+.6f}]",
        ))

    target, source = build_lever_arm_pair()
    for key in STRATEGIES:
        T = _align(target, source, key)
        roll_frac = _rotvec(T)[0] / -LEVER_ARM_ROLL
        headline[('lever-arm-50m', key)] = roll_frac
        rows.append((
            'lever-arm-50m', key, np.all(np.isfinite(T)),
            f"roll_recovered={roll_frac:+.6f}",
        ))

    with capsys.disabled():
        print("\n\n  fixture           strategy    finite  measurement")
        print("  " + "-" * 78)
        for fixture, key, finite, note in rows:
            print(f"  {fixture:<18}{key:<12}{str(bool(finite)):<8}{note}")
        print()

        for name, (tgt, src) in (
            ('well-conditioned', build_well_conditioned_pair()[:2]),
            ('flat-plane', build_flat_plane_pair()),
            ('near-degenerate', build_near_degenerate_pair()),
            ('lever-arm-50m', build_lever_arm_pair()),
        ):
            H = _hessian(tgt, src)
            deg6 = analyse_hessian(H)
            degd = analyse_hessian_decoupled(H, kappa_threshold=10.0)
            mask6 = ''.join('1' if b else '0' for b in deg6.degenerate_mask)
            maskd = ''.join('1' if b else '0' for b in degd.degenerate_mask)
            print(f"  {name:<18}6x6 mask={mask6} (thr {deg6.lambda_threshold:.4g})"
                  f"   decoupled mask={maskd} "
                  f"(ok={degd.factorization_ok}, cond_t={degd.cond_schur_t:.4g}, "
                  f"cond_R={degd.cond_schur_R:.4g})")
        print("  masks are [tx ty tz wx wy wz]\n")

    # Everything that ran, ran to a finite answer.
    for fixture, key, finite, note in rows:
        if 'raised' in note:
            continue
        assert finite, f"{fixture}/{key} produced a non-finite transform"

    # The three headline claims, re-asserted from the printed numbers against
    # the same named bands the dedicated tests use.  Duplicated deliberately:
    # this is what stops the table above from silently disagreeing with the
    # docstrings it is supposed to let a reader verify.
    sr_vertical = headline[('near-degenerate', 'C_sr_lm')][2]
    assert abs(sr_vertical) < SR_VERTICAL_RECOVERY_MAX, (
        "SR kept "
        f"{sr_vertical:.3%} of the well-observed vertical correction; "
        "TestNearDegenerateOffset.test_sr_also_suppresses_the_well_observed_axis "
        "documents essentially none"
    )

    sr_roll = headline[('lever-arm-50m', 'C_sr_lm')]
    assert abs(sr_roll) < SR_ROLL_RECOVERY_MAX, (
        f"SR kept {sr_roll:.3%} of the recoverable roll; "
        "TestLeverArmDetection.test_roll_is_recoverable_so_the_6x6_flag_is_a"
        "_false_positive documents essentially none"
    )

    lm_lateral = headline[('near-degenerate', 'B_lm')][0]
    lo, hi = LM_30_ITER_BAND
    assert lo < lm_lateral < hi, (
        f"LM recovered {lm_lateral:.4f} of the lateral offset in 30 "
        f"iterations, outside the documented band ({lo}, {hi}); "
        "TestNearDegenerateOffset.test_lm_shortfall_is_budget_not_bias "
        "reads that number as a partially-converged solve"
    )

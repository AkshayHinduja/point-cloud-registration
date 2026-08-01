"""
Unit tests for the decoupled (Schur-complement) degeneracy analysis:
``point_cloud_registration.degeneracy.analyse_hessian_decoupled``.

The method under test is a clean-room implementation of the detection and
characterization math published in

    Hu et al., "DCReg: Decoupled Characterization for Efficient Degenerate
    LiDAR Registration", IJRR 2026, arXiv:2509.06285.

Why it exists alongside :func:`analyse_hessian` (Zhang & Singh solution
remapping): the 6x6 spectrum mixes translational and rotational units, so
rotation eigenvalues carry a squared lever-arm (m^2) scaling that can hide a
rotational degeneracy behind translational curvature; and the 6x6 eigenvalue
test compares an eigenvalue (which carries the absolute scale of the problem)
against a unitless threshold, so it is *not* invariant to a uniform rescaling
of H.  The decoupled analysis tests each block against its own maximum, so the
criterion is a ratio-vs-ratio comparison and is scale invariant.

GOLDEN-VECTOR PROVENANCE
------------------------
The golden fixture reproduces the DCReg project's published minimal example,
``dcreg_minimal_example``, for one purpose only: to give this clean-room
implementation an oracle to cross-check against.  Both halves below are
*published facts about a synthetic problem* -- an input specification and the
output it is documented to produce -- not an implementation copy.  No reference
algorithm source was read, and none of the logic in
``point_cloud_registration.degeneracy`` derives from anything but the published
mathematics.

* Input constants -- the example's synthetic weak-axis system: ``basis`` is the
  6x6 identity with ``basis(0,2)=0.35``, ``basis(1,5)=0.15``,
  ``basis(2,5)=0.45``, ``basis(3,2)=0.20``, ``basis(4,5)=0.25``;
  ``stiffness = [6.0, 4.5, 1.1, 5.0, 3.5, 0.18]``;
  ``J = basis * diag(stiffness)``; ``H = J^T J``; run with
  ``degeneracy_condition_threshold = 10.0`` and ``kappa_target = 10.0``.

* Expected outputs -- ``DCReg/README.md`` lines 141-170, the
  "Representative Module-Level Log (dcreg_minimal_example)" block:

      cond_full: 1122.345689
      cond_schur_rot: 36.087707
      cond_schur_trans: 742.066396
      degenerate_mask: 001001
      raw_lambda_rot:  1.001796 19.881966 36.152505
      raw_lambda_trans:  0.032394 12.252030 24.038714
      aligned_lambda_rpy: 36.152505 19.881966  1.001796
      aligned_lambda_xyz: 24.038714 12.252030  0.032394
      r0 = 0.995659*roll + 0.000001*pitch + 0.004340*yaw
      r1 = 0.000004*roll + 0.999791*pitch + 0.000205*yaw
      r2 = 0.004337*roll + 0.000208*pitch + 0.995456*yaw
      t0 = 0.999989*x + 0.000000*y + 0.000011*z
      t1 = 0.000000*x + 0.999834*y + 0.000166*z
      t2 = 0.000011*x + 0.000166*y + 0.999823*z
      clamped_lambda_rpy: 36.152505 19.881966  3.615250
      clamped_lambda_xyz: 24.038714 12.252030  2.403871

DOF ORDER
---------
The reference example is written in the *reference* DOF order
``[roll, pitch, yaw, x, y, z]``.  This library uses ``[tx, ty, tz, wx, wy,
wz]`` (translation first), so the fixture permutes with ``p = [3, 4, 5, 0, 1,
2]`` before calling.  ``degenerate_mask: 001001`` (reference order) therefore
becomes PCR indices 2 (tz) and 5 (wz / yaw).

Tolerances: the reference log is printed with ``setprecision(6)`` fixed-point,
so a printed value carries up to 5e-7 of rounding error.  Comparisons use
``rtol=1e-5, atol=1e-6`` so that the small trans eigenvalue (0.032394), whose
print rounding exceeds its own 1e-5 relative tolerance, is still comparable.
"""
import numpy as np
import pytest

from point_cloud_registration.degeneracy import (
    DecoupledDegeneracyResult,
    analyse_hessian,
    analyse_hessian_decoupled,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

# Reference -> PCR DOF permutation: [roll,pitch,yaw,x,y,z] -> [tx,ty,tz,wx,wy,wz]
P_REF_TO_PCR = [3, 4, 5, 0, 1, 2]


def _golden_h_pcr() -> np.ndarray:
    """
    The ``dcreg_minimal_example`` synthetic weak-axis Hessian, permuted into
    PCR DOF order.  Input constants reproduced as published facts; see the
    module docstring for the provenance note.
    """
    basis = np.eye(6)
    basis[0, 2] = 0.35
    basis[1, 5] = 0.15
    basis[2, 5] = 0.45
    basis[3, 2] = 0.20
    basis[4, 5] = 0.25

    stiffness = np.array([6.0, 4.5, 1.1, 5.0, 3.5, 0.18])

    J = basis @ np.diag(stiffness)          # reference order [r, p, y, x, y, z]
    H_ref = J.T @ J
    return H_ref[np.ix_(P_REF_TO_PCR, P_REF_TO_PCR)]


# Expected outputs, transcribed from DCReg/README.md (see module docstring).
GOLDEN_COND_FULL = 1122.345689
GOLDEN_COND_SCHUR_R = 36.087707
GOLDEN_COND_SCHUR_T = 742.066396
GOLDEN_MASK_PCR = np.array([False, False, True, False, False, True])  # 001001 (ref order)
GOLDEN_RAW_LAMBDA_R = np.array([1.001796, 19.881966, 36.152505])
GOLDEN_RAW_LAMBDA_T = np.array([0.032394, 12.252030, 24.038714])
GOLDEN_ALIGNED_LAMBDA_R = np.array([36.152505, 19.881966, 1.001796])
GOLDEN_ALIGNED_LAMBDA_T = np.array([24.038714, 12.252030, 0.032394])
GOLDEN_CLAMPED_LAMBDA_R = np.array([36.152505, 19.881966, 3.615250])
GOLDEN_CLAMPED_LAMBDA_T = np.array([24.038714, 12.252030, 2.403871])
# ratios[axis, mode]: column m is mode m expressed as a mixture of physical axes
GOLDEN_CONTRIB_R = np.array([
    [0.995659, 0.000004, 0.004337],
    [0.000001, 0.999791, 0.000208],
    [0.004340, 0.000205, 0.995456],
])
GOLDEN_CONTRIB_T = np.array([
    [0.999989, 0.000000, 0.000011],
    [0.000000, 0.999834, 0.000166],
    [0.000011, 0.000166, 0.999823],
])

RTOL = 1e-5
ATOL = 1e-6


def _weak_translation_h() -> np.ndarray:
    """
    Hessian whose *diagonal* blocks are well conditioned but whose tx direction
    is nearly explained by the wx column -- i.e. a degeneracy that lives purely
    in the translation/rotation coupling.

    J columns (PCR order): tx = e0 + d*e3, ty = e4, tz = e5, wx = e0, wy = e1,
    wz = e2, so that

        H_tt = diag(1+d^2, 1, 1)          cond ~ 1
        H_ww = I                          cond = 1
        H_tw = [[1,0,0],[0,0,0],[0,0,0]]
        S_t  = diag(d^2, 1, 1)            cond = 1/d^2
    """
    d = 0.05
    J = np.zeros((6, 6))
    J[0, 0] = 1.0
    J[3, 0] = d          # tx column
    J[4, 1] = 1.0        # ty column
    J[5, 2] = 1.0        # tz column
    J[0, 3] = 1.0        # wx column
    J[1, 4] = 1.0        # wy column
    J[2, 5] = 1.0        # wz column
    return J.T @ J


# --------------------------------------------------------------------------
# 1 - Golden vector
# --------------------------------------------------------------------------

class TestGoldenVector:
    """Reproduces the reference implementation's published module-level log."""

    def setup_method(self):
        self.H = _golden_h_pcr()
        self.res = analyse_hessian_decoupled(
            self.H, kappa_threshold=10.0, kappa_target=10.0
        )

    def test_returns_result_dataclass(self):
        assert isinstance(self.res, DecoupledDegeneracyResult)
        assert self.res.factorization_ok

    def test_degenerate_mask_is_yaw_and_z(self):
        """README ``degenerate_mask: 001001`` -> PCR indices 2 (tz) and 5 (wz)."""
        np.testing.assert_array_equal(self.res.degenerate_mask, GOLDEN_MASK_PCR)
        assert self.res.is_degenerate
        assert self.res.num_constrained_dof == 4

    def test_raw_schur_eigenvalues(self):
        np.testing.assert_allclose(
            self.res.eigenvalues_R, GOLDEN_RAW_LAMBDA_R, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            self.res.eigenvalues_t, GOLDEN_RAW_LAMBDA_T, rtol=RTOL, atol=ATOL
        )

    def test_aligned_lambdas(self):
        np.testing.assert_allclose(
            self.res.aligned_lambda_R, GOLDEN_ALIGNED_LAMBDA_R, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            self.res.aligned_lambda_t, GOLDEN_ALIGNED_LAMBDA_T, rtol=RTOL, atol=ATOL
        )

    def test_clamped_lambdas(self):
        np.testing.assert_allclose(
            self.res.clamped_lambda_R, GOLDEN_CLAMPED_LAMBDA_R, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            self.res.clamped_lambda_t, GOLDEN_CLAMPED_LAMBDA_T, rtol=RTOL, atol=ATOL
        )

    def test_condition_numbers(self):
        assert self.res.cond_schur_R == pytest.approx(GOLDEN_COND_SCHUR_R, rel=RTOL)
        assert self.res.cond_schur_t == pytest.approx(GOLDEN_COND_SCHUR_T, rel=RTOL)
        assert self.res.cond_full == pytest.approx(GOLDEN_COND_FULL, rel=RTOL)

    def test_axis_contribution_ratios(self):
        np.testing.assert_allclose(
            self.res.contribution_R, GOLDEN_CONTRIB_R, rtol=1e-3, atol=1e-5
        )
        np.testing.assert_allclose(
            self.res.contribution_t, GOLDEN_CONTRIB_T, rtol=1e-3, atol=1e-5
        )

    def test_schur_complement_definitions(self):
        """S_t = H_tt - H_tw inv(H_ww) H_wt and S_R the mirror image."""
        H = 0.5 * (self.H + self.H.T)
        H_tt, H_tw, H_wt, H_ww = H[:3, :3], H[:3, 3:], H[3:, :3], H[3:, 3:]
        S_t = H_tt - H_tw @ np.linalg.inv(H_ww) @ H_wt
        S_R = H_ww - H_wt @ np.linalg.inv(H_tt) @ H_tw
        np.testing.assert_allclose(self.res.S_t, 0.5 * (S_t + S_t.T), atol=1e-12)
        np.testing.assert_allclose(self.res.S_R, 0.5 * (S_R + S_R.T), atol=1e-12)

    def test_kappa_target_defaults_to_threshold(self):
        default = analyse_hessian_decoupled(self.H, kappa_threshold=10.0)
        np.testing.assert_allclose(
            default.clamped_lambda_R, self.res.clamped_lambda_R, rtol=1e-12
        )
        np.testing.assert_allclose(
            default.clamped_lambda_t, self.res.clamped_lambda_t, rtol=1e-12
        )


# --------------------------------------------------------------------------
# 2 - Scale invariance
# --------------------------------------------------------------------------

class TestScaleInvariance:
    def test_mask_and_ratios_invariant_clamped_values_scale(self):
        H = _golden_h_pcr()
        base = analyse_hessian_decoupled(H, kappa_threshold=10.0, kappa_target=10.0)

        for c in (1e-4, 1.0, 1e4):
            res = analyse_hessian_decoupled(
                c * H, kappa_threshold=10.0, kappa_target=10.0
            )
            np.testing.assert_array_equal(res.degenerate_mask, base.degenerate_mask)
            # aligned-lambda ratios (within block) identical
            np.testing.assert_allclose(
                res.aligned_lambda_t / res.aligned_lambda_t[0],
                base.aligned_lambda_t / base.aligned_lambda_t[0],
                rtol=1e-9,
            )
            np.testing.assert_allclose(
                res.aligned_lambda_R / res.aligned_lambda_R[0],
                base.aligned_lambda_R / base.aligned_lambda_R[0],
                rtol=1e-9,
            )
            # clamped values scale linearly with c
            np.testing.assert_allclose(
                res.clamped_lambda_t, c * base.clamped_lambda_t, rtol=1e-9
            )
            np.testing.assert_allclose(
                res.clamped_lambda_R, c * base.clamped_lambda_R, rtol=1e-9
            )
            # block condition numbers unchanged
            assert res.cond_schur_t == pytest.approx(base.cond_schur_t, rel=1e-9)
            assert res.cond_schur_R == pytest.approx(base.cond_schur_R, rel=1e-9)

    def test_isotropic_scaling_flips_old_criterion_but_not_the_new_one(self):
        """
        Comparative pin: the 6x6 eigenvalue-vs-condition-number criterion of
        :func:`analyse_hessian` classifies two *identically conditioned*
        Hessians differently -- 0.5*I is called fully degenerate, 2*I fully
        constrained -- because it compares an eigenvalue against a unitless
        ratio.  The decoupled criterion compares ratios against ratios and
        calls both fully constrained.
        """
        small = analyse_hessian(0.5 * np.eye(6))
        large = analyse_hessian(2.0 * np.eye(6))
        assert small.degenerate_mask.all()
        assert not large.degenerate_mask.any()

        d_small = analyse_hessian_decoupled(0.5 * np.eye(6))
        d_large = analyse_hessian_decoupled(2.0 * np.eye(6))
        assert not d_small.degenerate_mask.any()
        assert not d_large.degenerate_mask.any()
        assert d_small.num_constrained_dof == 6
        assert d_large.num_constrained_dof == 6


# --------------------------------------------------------------------------
# 3 - Coupling reveals hidden degeneracy
# --------------------------------------------------------------------------

class TestCouplingRevealsDegeneracy:
    def test_schur_flags_what_diagonal_block_alone_misses(self):
        H = _weak_translation_h()
        res = analyse_hessian_decoupled(H, kappa_threshold=10.0)
        assert res.factorization_ok

        # The naive per-block test on H_tt alone sees nothing wrong.
        lam_tt = np.linalg.eigvalsh(H[:3, :3])
        assert lam_tt.max() / lam_tt.min() < 10.0, (
            "fixture broken: H_tt is supposed to look well conditioned"
        )
        naive_mask = (lam_tt.max() / np.maximum(lam_tt, 1e-12)) > 10.0
        assert not naive_mask.any()

        # The Schur complement exposes the coupling-induced weak direction.
        assert res.degenerate_mask[0], "tx should be flagged by the Schur test"
        assert res.cond_schur_t > 100.0
        np.testing.assert_allclose(
            res.S_t, np.diag([0.05 ** 2, 1.0, 1.0]), atol=1e-12
        )
        # ty / tz remain constrained
        assert not res.degenerate_mask[1]
        assert not res.degenerate_mask[2]

    def test_clamping_raises_only_the_flagged_axis(self):
        H = _weak_translation_h()
        res = analyse_hessian_decoupled(H, kappa_threshold=10.0, kappa_target=10.0)
        lam_max_t = res.eigenvalues_t.max()
        np.testing.assert_allclose(res.clamped_lambda_t[0], lam_max_t / 10.0, rtol=1e-9)
        np.testing.assert_allclose(
            res.clamped_lambda_t[1:], res.aligned_lambda_t[1:], rtol=1e-12
        )
        assert res.clamped_lambda_t[0] > res.aligned_lambda_t[0]


# --------------------------------------------------------------------------
# 4 - Isotropic, well-conditioned H
# --------------------------------------------------------------------------

class TestWellConditioned:
    def test_nothing_flagged_and_clamping_is_identity(self):
        H = np.diag([5.0, 5.0, 5.0, 4.0, 4.0, 4.0])
        res = analyse_hessian_decoupled(H)

        assert res.factorization_ok
        assert not res.is_degenerate
        assert not res.degenerate_mask.any()
        assert res.num_constrained_dof == 6

        np.testing.assert_allclose(res.eigenvalues_t, [5.0, 5.0, 5.0], atol=1e-12)
        np.testing.assert_allclose(res.eigenvalues_R, [4.0, 4.0, 4.0], atol=1e-12)
        np.testing.assert_allclose(res.aligned_lambda_t, res.clamped_lambda_t, atol=1e-12)
        np.testing.assert_allclose(res.aligned_lambda_R, res.clamped_lambda_R, atol=1e-12)
        np.testing.assert_allclose(res.clamped_lambda_t, [5.0, 5.0, 5.0], atol=1e-12)
        np.testing.assert_allclose(res.clamped_lambda_R, [4.0, 4.0, 4.0], atol=1e-12)
        assert res.cond_schur_t == pytest.approx(1.0)
        assert res.cond_schur_R == pytest.approx(1.0)
        assert res.cond_full == pytest.approx(5.0 / 4.0)


# --------------------------------------------------------------------------
# 5 - Singular block gate
# --------------------------------------------------------------------------

class TestSingularBlockGate:
    def test_zero_rotation_block_is_reported_not_raised(self):
        H = np.diag([5.0, 5.0, 5.0, 0.0, 0.0, 0.0])
        res = analyse_hessian_decoupled(H)

        assert res.factorization_ok is False
        assert res.degenerate_mask.all()
        assert res.is_degenerate
        assert res.num_constrained_dof == 0
        np.testing.assert_allclose(res.clamped_lambda_t, np.zeros(3), atol=0.0)
        np.testing.assert_allclose(res.clamped_lambda_R, np.zeros(3), atol=0.0)
        assert res.S_t.shape == (3, 3)
        assert res.S_R.shape == (3, 3)
        assert res.eigenvalues_t.shape == (3,)
        assert res.eigenvectors_R.shape == (3, 3)
        assert res.aligned_basis_t.shape == (3, 3)
        assert res.contribution_R.shape == (3, 3)

    def test_zero_translation_block_also_gated(self):
        H = np.diag([0.0, 0.0, 0.0, 4.0, 4.0, 4.0])
        res = analyse_hessian_decoupled(H)
        assert res.factorization_ok is False
        assert res.degenerate_mask.all()
        assert res.num_constrained_dof == 0


# --------------------------------------------------------------------------
# 6 - Alignment invariance
# --------------------------------------------------------------------------

class TestAlignmentInvariance:
    """
    The greedy signed permutation is a *relabelling* for reporting.  Any
    reconstruction of the form ``V f(Lambda) V^T`` must be identical whether it
    is assembled from the raw eigenpairs or from the aligned ones.
    """

    @staticmethod
    def _seeded_spd() -> np.ndarray:
        rng = np.random.default_rng(7)
        J = rng.standard_normal((40, 6))
        J[:, 2] *= 0.05          # deliberately weak tz so clamping is exercised
        return J.T @ J

    def test_inverse_reconstruction_matches_raw_ordering(self):
        H = self._seeded_spd()
        res = analyse_hessian_decoupled(H, kappa_threshold=10.0, kappa_target=10.0)
        assert res.factorization_ok
        assert res.is_degenerate, "fixture should flag at least one axis"

        for V, aligned_basis, clamped in (
            (res.eigenvectors_t, res.aligned_basis_t, res.clamped_lambda_t),
            (res.eigenvectors_R, res.aligned_basis_R, res.clamped_lambda_R),
        ):
            # Recover the permutation: aligned column a is +/- raw column perm[a].
            overlap = np.abs(aligned_basis.T @ V)     # (axis, raw_col)
            perm = np.argmax(overlap, axis=1)
            assert sorted(perm.tolist()) == [0, 1, 2], "alignment is not a permutation"
            np.testing.assert_allclose(overlap[np.arange(3), perm], 1.0, atol=1e-9)

            clamped_raw = np.empty(3)
            clamped_raw[perm] = clamped

            lhs = aligned_basis @ np.diag(1.0 / clamped) @ aligned_basis.T
            rhs = V @ np.diag(1.0 / clamped_raw) @ V.T
            np.testing.assert_allclose(lhs, rhs, atol=1e-9)

    def test_aligned_lambda_is_the_permuted_raw_lambda(self):
        H = self._seeded_spd()
        res = analyse_hessian_decoupled(H)
        np.testing.assert_allclose(
            np.sort(res.aligned_lambda_t), res.eigenvalues_t, rtol=1e-12
        )
        np.testing.assert_allclose(
            np.sort(res.aligned_lambda_R), res.eigenvalues_R, rtol=1e-12
        )


# --------------------------------------------------------------------------
# 7 - Contribution ratios / basis structure
# --------------------------------------------------------------------------

class TestContributionStructure:
    @staticmethod
    def _cases():
        rng = np.random.default_rng(11)
        A = rng.standard_normal((12, 6))
        return [
            _golden_h_pcr(),
            _weak_translation_h(),
            A.T @ A + 0.1 * np.eye(6),
        ]

    def test_aligned_basis_columns_are_unit_norm(self):
        for H in self._cases():
            res = analyse_hessian_decoupled(H)
            for B in (res.aligned_basis_t, res.aligned_basis_R):
                np.testing.assert_allclose(
                    np.linalg.norm(B, axis=0), np.ones(3), atol=1e-12
                )

    def test_contribution_columns_sum_to_one(self):
        for H in self._cases():
            res = analyse_hessian_decoupled(H)
            for C in (res.contribution_t, res.contribution_R):
                np.testing.assert_allclose(C.sum(axis=0), np.ones(3), atol=1e-12)
                assert (C >= 0.0).all()

    def test_contribution_is_squared_aligned_basis(self):
        for H in self._cases():
            res = analyse_hessian_decoupled(H)
            np.testing.assert_allclose(
                res.contribution_t, res.aligned_basis_t ** 2, atol=1e-15
            )
            np.testing.assert_allclose(
                res.contribution_R, res.aligned_basis_R ** 2, atol=1e-15
            )

    def test_aligned_basis_diagonal_is_non_negative(self):
        """Sign convention: aligned_basis[a, a] >= 0 after the greedy flip."""
        for H in self._cases():
            res = analyse_hessian_decoupled(H)
            for B in (res.aligned_basis_t, res.aligned_basis_R):
                assert (np.diag(B) >= 0.0).all()

    def test_aligned_basis_is_orthonormal(self):
        for H in self._cases():
            res = analyse_hessian_decoupled(H)
            for B in (res.aligned_basis_t, res.aligned_basis_R):
                np.testing.assert_allclose(B.T @ B, np.eye(3), atol=1e-12)


# --------------------------------------------------------------------------
# Package export
# --------------------------------------------------------------------------

def test_exported_from_package_root():
    import point_cloud_registration as pcr

    assert hasattr(pcr, "analyse_hessian_decoupled")
    assert hasattr(pcr, "DecoupledDegeneracyResult")

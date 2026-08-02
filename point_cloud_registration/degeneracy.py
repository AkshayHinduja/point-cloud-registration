"""
Degeneracy detection and solution remapping for point cloud registration.

Least-squares registration (ICP and its variants) takes a Gauss-Newton step
``dx = -H⁻¹ g`` at every iteration, where ``H`` is the 6×6 approximate Hessian
(``JᵀJ``) accumulated over the correspondences and ``g`` is the gradient.  When
the scene is geometrically under-determined — a flat plane, a straight
corridor, a surface of revolution — ``H`` is rank-deficient or ill-conditioned
along one or more directions, and the unconstrained solve slides the estimate
along those directions driven by nothing but noise.

This module implements *solution remapping*: eigendecompose ``H``, classify each
eigen-direction as constrained or degenerate, and project the update so that its
component along every degenerate direction is zero.  The estimate is then only
updated where the data actually supports it, and stays put elsewhere.

Scope: :func:`analyse_hessian` *classifies* directions and copes with an
exactly rank-deficient ``H``; :func:`apply_sr_solve` still inverts ``H`` to form
the step, so it requires an invertible one.  Damping a singular Hessian is the
caller's responsibility — see :func:`apply_sr_solve` for the contract.

A second, independent analysis lives here as well:
:func:`analyse_hessian_decoupled` splits ``H`` into its translation and rotation
blocks via Schur complements and classifies each block against its own
spectrum.  It reports *which physical axis* is unobservable rather than which
abstract eigen-direction, and its criterion is invariant to a uniform rescaling
of ``H`` (which :func:`analyse_hessian`'s is not).  The two are separate entry
points; nothing above changes.

:func:`dcreg_solve` is the matching *mitigation* step for that second analysis,
with two modes.  ``'pcg'`` is the published solver: a preconditioned CG on the
unmodified normal equations, which at full convergence returns the plain
Gauss-Newton step — it improves conditioning, it does not move the minimum.
``'clamped'`` adds the per-block spectral deficit along the flagged axes only,
and does move the minimum.  Read that function's docstring before picking one;
the difference between them is the difference between a numerical convenience
and a change of estimate.

References:
    A. Hinduja, B.-J. Ho and M. Kaess, "Degeneracy-Aware Factors with
    Applications to Underwater SLAM", IEEE/RSJ International Conference on
    Intelligent Robots and Systems (IROS), 2019, pp. 1293-1299,
    doi: 10.1109/IROS40897.2019.8968577 — introduced solution remapping
    inside the ICP iteration itself (its Algorithm 1 is the procedure
    implemented by :func:`analyse_hessian` + :func:`apply_sr_solve`,
    including the ``sqrt(lambda_max / lambda_min)`` threshold).

    J. Zhang, M. Kaess and S. Singh, "On Degeneracy of Optimization-based
    State Estimation Problems", IEEE International Conference on Robotics
    and Automation (ICRA), 2016 — the origin of the solution-remapping
    update for optimization-based state estimation.

    Hu et al., "DCReg: Decoupled Characterization for Efficient Degenerate
    LiDAR Registration", International Journal of Robotics Research (IJRR),
    2026 — arXiv:2509.06285.  (Source of the decoupled Schur analysis; this is
    an independent implementation written from the published mathematics.)

DOF order
---------
All 6-vectors and 6×6 matrices here use this library's tangent-space ordering::

    [tx, ty, tz, ωx, ωy, ωz]

i.e. translation first, rotation second — the ordering consumed by
``math_tools.plus()`` and produced by the ``calc_H_g_e2()`` methods of the
registration classes.  Callers that work in a different convention (for example
a factor-graph library that orders rotation first) must permute ``H`` and ``g``
before calling and permute the returned step back afterwards.

This module depends on NumPy only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DegeneracyResult:
    """
    Output of analyse_hessian().

    Attributes
    ----------
    eigenvalues:
        Ascending eigenvalues of H.
    eigenvectors:
        Corresponding eigenvectors as *columns*, so ``eigenvectors[:, i]`` is the
        unit direction whose curvature is ``eigenvalues[i]``.
    condition_number:
        ``sqrt(λ_max / λ_min)``; also the adaptive degeneracy threshold when no
        explicit threshold was supplied.  ``inf`` for a rank-zero Hessian.
    lambda_threshold:
        The eigenvalue threshold actually applied.
    degenerate_mask:
        ``True`` where the direction is degenerate (data does not constrain it).
    num_constrained_dof:
        Number of directions the data does constrain, 0-6.
    is_degenerate:
        ``True`` if at least one direction is degenerate.
    V_constrained:
        The constrained eigenvectors as unit-norm *rows*.
    """
    eigenvalues: np.ndarray       # (6,) ascending, in PCR order
    eigenvectors: np.ndarray      # (6,6) columns are eigenvectors, in PCR order
    condition_number: float       # sqrt(λ_max / λ_min)
    lambda_threshold: float       # eigenvalue threshold used
    degenerate_mask: np.ndarray   # (6,) bool, True = degenerate direction
    num_constrained_dof: int      # count of non-degenerate directions
    is_degenerate: bool           # True if any DOF is degenerate
    # Non-degenerate eigenvector rows in PCR order, shape (num_constrained_dof, 6).
    # Empty (0, 6) array when all directions are degenerate.
    V_constrained: np.ndarray


def analyse_hessian(
    H: np.ndarray,
    lambda_threshold: Optional[float] = None,
) -> DegeneracyResult:
    """
    Eigendecompose a 6×6 registration Hessian (ATA) and identify degenerate
    directions.

    H must be in this library's DOF order: [tx, ty, tz, ωx, ωy, ωz].

    Criterion
    ---------
    A direction is degenerate when its eigenvalue falls below the threshold

        lambda_cn = sqrt(lambda_max / lambda_min)

    the square root of the Hessian's condition number.  Note what that compares:
    an eigenvalue, which carries the scale of the problem (correspondence count,
    units of the residual), against a *unitless* ratio.  Classification therefore
    depends on the absolute magnitude of H and not on its conditioning alone —
    scaling H uniformly moves the eigenvalues but leaves lambda_cn untouched.
    ``H = 2*I`` and ``H = 0.5*I`` are both perfectly conditioned (lambda_cn = 1),
    yet the first has all six directions constrained and the second has all six
    degenerate.  Concretely, the best-observed direction survives only when
    ``lambda_max >= sqrt(lambda_max / lambda_min)``, i.e. when
    ``lambda_max * lambda_min >= 1``.

    This is the criterion as published and as implemented in the reference
    solution-remapping code, and it is kept here unchanged for fidelity to it.
    Callers who have a calibrated absolute curvature floor for their sensor and
    geometry should use the lambda_threshold override instead of relying on it.

    Structural zeros are handled in two phases so that exact rank deficiency does
    not swamp the test:

    1. Eigenvalues below ``1e-10 * lambda_max`` are treated as structural zeros
       (a genuinely unobserved direction: too few correspondences, or a scene
       with an exact continuous symmetry).  They are always degenerate.
    2. lambda_cn is then formed from the smallest *non*-structural-zero
       eigenvalue.  Without this, a single exact-zero eigenvalue would send the
       condition number to infinity and mark every direction — including the
       well-observed ones — degenerate.

    A Hessian whose largest eigenvalue is itself below ``1e-10`` carries no
    information at all (e.g. zero correspondences); every direction is reported
    degenerate and ``V_constrained`` is an empty ``(0, 6)`` array.

    This classification is well defined for a singular H, but classifying is not
    solving: :func:`apply_sr_solve` inverts H and will raise on an exactly
    singular one.  See its docstring for how to pair the two.

    If lambda_threshold is not None it overrides the adaptive threshold with a
    fixed absolute eigenvalue floor (useful when a caller has a calibrated
    minimum curvature for its sensor and geometry).  The reported condition
    number is still computed, with lambda_min floored at ``1e-12`` to keep it
    finite.
    """
    H = np.asarray(H, dtype=float)
    assert H.shape == (6, 6), f"Expected 6×6 Hessian, got {H.shape}"

    eigenvalues, eigenvectors = np.linalg.eigh(H)  # ascending order
    lam_max = float(eigenvalues[-1])

    # Zero or near-zero Hessian → no correspondences → all directions degenerate.
    if lam_max < 1e-10:
        return DegeneracyResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            condition_number=float('inf'),
            lambda_threshold=float('inf'),
            degenerate_mask=np.ones(6, dtype=bool),
            num_constrained_dof=0,
            is_degenerate=True,
            V_constrained=np.empty((0, 6)),
        )

    if lambda_threshold is not None:
        # User-supplied override (e.g. from SR solve fixed threshold).
        lam_min = float(max(eigenvalues[0], 1e-12))
        condition_number = float(np.sqrt(lam_max / lam_min))
        threshold = float(lambda_threshold)
        degenerate_mask = eigenvalues < threshold
    else:
        # Two-phase: structural zeros first, then the condition-number test
        # among the non-zeros.  Rank-deficient Hessians (flat surface,
        # single-point) have exact-zero eigenvalues that must not inflate the
        # condition number to infinity.
        abs_zero_thresh = 1e-10 * lam_max
        struct_zero_mask = eigenvalues < abs_zero_thresh
        if struct_zero_mask.all():
            condition_number = float('inf')
            threshold = float('inf')
            degenerate_mask = np.ones(6, dtype=bool)
        else:
            lam_min_pos = float(eigenvalues[~struct_zero_mask][0])
            condition_number = float(np.sqrt(lam_max / lam_min_pos))
            threshold = condition_number
            degenerate_mask = eigenvalues < threshold

    constrained_indices = np.where(~degenerate_mask)[0]
    num_constrained = int(len(constrained_indices))

    # V_constrained: rows are non-degenerate eigenvectors in PCR order
    V_constrained = eigenvectors[:, constrained_indices].T  # (num_constrained, 6)

    return DegeneracyResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        condition_number=condition_number,
        lambda_threshold=threshold,
        degenerate_mask=degenerate_mask,
        num_constrained_dof=num_constrained,
        is_degenerate=bool(np.any(degenerate_mask)),
        V_constrained=V_constrained,
    )


def apply_sr_solve(
    H: np.ndarray,
    g: np.ndarray,
    deg: DegeneracyResult,
) -> np.ndarray:
    """
    Apply solution-remapping to the registration linear solve.

    Standard solve:  dx = -H⁻¹ g
    SR solve:        dx = -Vf_inv @ Vu_filtered @ H⁻¹ @ g

    ``Vu_filtered`` is the matrix of eigenvector rows with the degenerate rows
    replaced by zero, and ``Vf_inv`` maps back from the eigenbasis to the DOF
    basis.  The composition projects the Gauss-Newton step onto the span of the
    constrained eigen-directions: the optimizer still takes the full step where
    the data supports it, and no step at all along the ill-conditioned DOFs.

    When nothing is degenerate this reduces exactly to the standard solve.

    Contract: H must be invertible.  Detecting degeneracy does not remove the
    inversion — ``np.linalg.solve(H, g)`` runs on the H you pass, before any
    projection, so an exactly singular H raises ``np.linalg.LinAlgError`` no
    matter what ``deg`` says about it.  A caller whose raw Hessian can be exactly
    rank-deficient (a plane fitted to coplanar points, fewer correspondences than
    DOFs) should pass a regularized ``H + lambda*I`` here while taking ``deg``
    from :func:`analyse_hessian` on the **raw** H.  An isotropic shift adds
    lambda to every eigenvalue and leaves the eigenvectors untouched, so the
    projection this function applies is unchanged; only the step length along the
    constrained directions is damped, and analysing the raw H keeps the
    degeneracy classification honest (damping before analysis raises lambda_min
    and so lowers the threshold; for a large enough lambda the weak directions
    stop being flagged at all).

    Args:
        H:   6×6 Hessian from calc_H_g_e2 (in PCR order), invertible — see Contract
        g:   6-element gradient from calc_H_g_e2
        deg: DegeneracyResult from analyse_hessian(H)

    Returns:
        6-element step dx in PCR order

    Raises:
        numpy.linalg.LinAlgError: if H (or the eigenvector matrix) is singular.
    """
    Vu = deg.eigenvectors.T.copy()          # (6, 6) rows = eigenvectors
    for i in range(6):
        if deg.degenerate_mask[i]:
            Vu[i, :] = 0.0                  # zero degenerate rows

    Vf_inv = np.linalg.inv(deg.eigenvectors.T)
    H_inv_g = np.linalg.solve(H, g)
    return -(Vf_inv @ Vu @ H_inv_g)


# ===========================================================================
# Decoupled (Schur-complement) degeneracy analysis
# ===========================================================================
#
# Independent implementation of the detection/characterization mathematics of
#
#     Hu et al., "DCReg: Decoupled Characterization for Efficient Degenerate
#     LiDAR Registration", IJRR 2026, arXiv:2509.06285.
#
# Written from the published equations only.

# A block whose 2-norm condition number exceeds this is treated as singular for
# the purpose of forming a Schur complement.
_SINGULAR_BLOCK_COND = 1e12

# Floor applied to a block eigenvalue before dividing by it, so a structurally
# zero (or slightly negative, from round-off) eigenvalue cannot produce inf/nan.
_LAMBDA_FLOOR = 1e-12

# Floor on a clamped eigenvalue, so downstream inverses stay finite even if a
# whole block is numerically dead.
_CLAMP_FLOOR = 1e-9


@dataclass
class DecoupledDegeneracyResult:
    """
    Output of :func:`analyse_hessian_decoupled`.

    All 3-vectors and 3×3 matrices are per block: ``*_t`` is the translation
    block, indexed ``[x, y, z]``, and ``*_R`` is the rotation block, indexed
    ``[ωx (roll), ωy (pitch), ωz (yaw)]``.  ``degenerate_mask`` is the only
    6-vector and follows this library's DOF order ``[tx, ty, tz, ωx, ωy, ωz]``.

    Attributes
    ----------
    S_t, S_R:
        The Schur complements ``H_tt - H_tw inv(H_ww) H_wt`` and
        ``H_ww - H_wt inv(H_tt) H_tw``, symmetrized.  ``S_t`` is the curvature
        of the translation block *after* the rotation DOFs have been optimally
        eliminated: the observability that translation actually owns.
    eigenvalues_t, eigenvalues_R:
        Ascending eigenvalues of the corresponding Schur complement (raw, as
        returned by ``eigh``).
    eigenvectors_t, eigenvectors_R:
        The matching eigenvectors as *columns*, raw (same order as the
        eigenvalues).
    aligned_lambda_t, aligned_lambda_R:
        The same eigenvalues, relabelled per physical axis: entry ``a`` is the
        eigenvalue of the eigen-direction most aligned with axis ``a``.
    aligned_basis_t, aligned_basis_R:
        The matching eigenvectors as columns, permuted and sign-flipped so that
        column ``a`` is the mode assigned to axis ``a`` and its ``a``-th entry
        is non-negative.
    contribution_t, contribution_R:
        ``aligned_basis ** 2`` elementwise.  Column ``a`` decomposes mode ``a``
        into its physical-axis content and sums to 1; a column with a dominant
        entry means the mode *is* that axis, a spread-out column means the mode
        is a mixture and the per-axis label is only nominal.
    degenerate_mask:
        ``True`` where the axis is degenerate, in DOF order
        ``[tx, ty, tz, ωx, ωy, ωz]``.
    clamped_lambda_t, clamped_lambda_R:
        ``aligned_lambda`` with each flagged axis raised to
        ``max(its own eigenvalue, lambda_max_of_block / kappa_target)``;
        unflagged axes are untouched.  This is the spectrum a mitigation step
        (preconditioning / regularized solve) should use in place of the raw
        one.  Clamping is *monotone*: ``clamped_lambda >= aligned_lambda``
        elementwise for any positive ``kappa_target``, so the spectral deficit
        ``clamped - aligned`` is a valid PSD regularizer.
    cond_schur_t, cond_schur_R:
        ``lambda_max / lambda_min`` of the corresponding Schur complement.
    cond_full:
        ``lambda_max / lambda_min`` of the full 6×6 ``H``.  Diagnostic only —
        it mixes translational and rotational units and is exactly the quantity
        the decoupled analysis exists to avoid trusting.
    factorization_ok:
        ``False`` if ``H_tt`` or ``H_ww`` was too ill-conditioned to invert, in
        which case no Schur complement exists, every axis is reported
        degenerate, and the Schur/eigen fields are zero-filled.
    is_degenerate:
        ``any(degenerate_mask)``.
    num_constrained_dof:
        ``6 - sum(degenerate_mask)``.
    """
    S_t: np.ndarray                 # (3,3)
    S_R: np.ndarray                 # (3,3)
    eigenvalues_t: np.ndarray       # (3,) ascending, raw
    eigenvalues_R: np.ndarray       # (3,) ascending, raw
    eigenvectors_t: np.ndarray      # (3,3) columns, raw
    eigenvectors_R: np.ndarray      # (3,3) columns, raw
    aligned_lambda_t: np.ndarray    # (3,) per axis [x, y, z]
    aligned_lambda_R: np.ndarray    # (3,) per axis [ωx, ωy, ωz]
    aligned_basis_t: np.ndarray     # (3,3) signed-permuted columns
    aligned_basis_R: np.ndarray     # (3,3)
    contribution_t: np.ndarray      # (3,3) aligned_basis_t ** 2
    contribution_R: np.ndarray      # (3,3)
    degenerate_mask: np.ndarray     # (6,) bool, DOF order
    clamped_lambda_t: np.ndarray    # (3,)
    clamped_lambda_R: np.ndarray    # (3,)
    cond_schur_t: float
    cond_schur_R: float
    cond_full: float                # diagnostic only
    factorization_ok: bool
    is_degenerate: bool
    num_constrained_dof: int


def _block_is_invertible(block: np.ndarray) -> bool:
    """
    True when ``block`` can be inverted well enough to form a Schur complement.

    Uses singular values rather than ``np.linalg.cond`` so that an exactly zero
    block yields a clean ``False`` instead of a 0/0 RuntimeWarning.
    """
    try:
        s = np.linalg.svd(block, compute_uv=False)
    except np.linalg.LinAlgError:
        return False
    if not np.all(np.isfinite(s)):
        return False
    if s[-1] <= 0.0:
        return False
    return bool(s[0] / s[-1] <= _SINGULAR_BLOCK_COND)


def _align_to_axes(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> tuple:
    """
    Relabel an eigendecomposition per physical axis (greedy signed permutation).

    For axis ``a`` in order 0, 1, 2: among the eigenvector columns not yet
    claimed, take the one with the largest ``|V[a, col]|`` — the mode that
    points most along axis ``a`` — and flip its sign so ``V[a, col] >= 0``.

    This is a *relabelling for reporting only*.  Permuting eigenpairs (and
    flipping eigenvector signs) leaves every spectral reconstruction
    ``V f(Λ) Vᵀ = Σ_i f(λ_i) v_i v_iᵀ`` unchanged, because the sum is over the
    same set of (eigenvalue, rank-1 projector) pairs and ``(-v)(-v)ᵀ = v vᵀ``.
    So a caller may build a preconditioner or a covariance from the aligned
    quantities and get bit-comparable results to the raw ones.

    Returns ``(aligned_lambda, aligned_basis)``.
    """
    n = eigenvectors.shape[0]
    used = np.zeros(n, dtype=bool)
    aligned_lambda = np.empty(n, dtype=float)
    aligned_basis = np.empty((n, n), dtype=float)

    for axis in range(n):
        candidates = np.flatnonzero(~used)
        chosen = int(candidates[int(np.argmax(np.abs(eigenvectors[axis, candidates])))])
        used[chosen] = True
        column = eigenvectors[:, chosen]
        if column[axis] < 0.0:
            column = -column
        aligned_basis[:, axis] = column
        aligned_lambda[axis] = eigenvalues[chosen]

    return aligned_lambda, aligned_basis


def _ratio(lam_max: float, lam_min: float) -> float:
    """``lam_max / max(lam_min, floor)``; ``inf`` for a numerically dead block."""
    if lam_max <= _LAMBDA_FLOOR:
        return float('inf')
    return float(lam_max / max(lam_min, _LAMBDA_FLOOR))


def analyse_hessian_decoupled(
    H: np.ndarray,
    kappa_threshold: float = 10.0,
    kappa_target: Optional[float] = None,
) -> DecoupledDegeneracyResult:
    """
    Decoupled (Schur-complement) degeneracy analysis of a 6×6 registration
    Hessian, reporting degeneracy *per physical axis* per block.

    Implements the detection and characterization mathematics of

        Hu et al., "DCReg: Decoupled Characterization for Efficient Degenerate
        LiDAR Registration", IJRR 2026, arXiv:2509.06285.

    ``H`` must be in this library's DOF order ``[tx, ty, tz, ωx, ωy, ωz]``; the
    translation and rotation blocks are read off directly, no permutation.

    Why the Schur complement, and not ``eigh(H)``
    ---------------------------------------------
    Two separate problems with eigendecomposing the full 6×6 matrix:

    1. **Scale disparity.**  The rotation columns of the Jacobian are lever-arm
       terms ``n × (R p)``, so rotational curvature carries an extra length²
       factor relative to translational curvature.  The six eigenvalues of ``H``
       therefore do not live in the same units, and their spread is dominated by
       the point-cloud extent rather than by observability.  With a cloud tens
       of metres across, a *badly* observed rotation axis can still out-scale a
       *well* observed translation axis, and a single 6×6 ratio test silently
       hides it.  Testing each block against its own maximum removes the common
       factor.

    2. **Coupling.**  ``H_tt`` alone is the curvature of translation *with the
       rotation held fixed*, which flatters it: it ignores that some of that
       curvature is shared with rotation and cannot be attributed to
       translation.  The Schur complement
       ``S_t = H_tt - H_tw inv(H_ww) H_wt`` is the curvature that survives after
       rotation has been optimally eliminated, and for positive definite ``H``
       it satisfies ``S_t ⪯ H_tt`` — coupling can only *reduce* observability,
       never add to it.  A direction that looks well constrained in ``H_tt`` and
       collapses in ``S_t`` is one whose apparent observability was really the
       rotation's.

    Criterion
    ---------
    Axis ``a`` of block ``B`` is degenerate when

        lambda_max(B) / lambda_a(B) > kappa_threshold

    i.e. an eigenvalue *ratio* against a ratio threshold.  Both sides are
    unitless, so multiplying ``H`` by any positive constant leaves the mask
    exactly unchanged.  Contrast :func:`analyse_hessian`, which compares an
    eigenvalue against ``sqrt(lambda_max / lambda_min)`` and therefore does
    depend on the absolute magnitude of ``H`` (see its docstring: ``0.5*I`` is
    called fully degenerate while ``2*I`` is called fully constrained, though
    both are perfectly conditioned).  Scale invariance matters here because the
    magnitude of ``H`` tracks correspondence count and residual weighting, which
    are properties of the run, not of the geometry.

    The two blocks are tested independently: the translation mask never depends
    on the rotation spectrum, which is the whole point of decoupling.

    Frame
    -----
    The ``x/y/z`` and ``roll/pitch/yaw`` labels are meaningful only in the frame
    whose Jacobian produced ``H`` — for the registration classes in this package
    that is the source-cloud (body) frame, since ``calc_H_g_e2()`` differentiates
    with respect to a body-frame perturbation.  "z is degenerate" therefore means
    the sensor's z, not the world's.  Callers reporting axes to a user in another
    frame must rotate ``aligned_basis`` themselves; the per-axis labelling is not
    frame agnostic.

    Args:
        H: 6×6 approximate Hessian (JᵀJ) in DOF order [tx, ty, tz, ωx, ωy, ωz].
           Symmetrized on entry.
        kappa_threshold: Per-block eigenvalue-ratio threshold above which an
           axis is called degenerate.  Must be positive.
        kappa_target: Target ratio used when clamping the flagged eigenvalues.
           Defaults to ``kappa_threshold``, which makes clamping the smallest
           raise that brings the flagged axis just inside the threshold.  Must be
           positive.  A value below ``kappa_threshold`` clamps harder (raises
           more).  A value above it is a gentler nudge: the target level
           ``lambda_max / kappa_target`` drops below the eigenvalue of any
           flagged axis whose ratio lies between the two thresholds, and because
           clamping is monotone (``max`` against the axis's own eigenvalue) such
           an axis simply keeps what it has rather than being lowered.  Flagged
           axes further out than ``kappa_target`` are still raised.

    Returns:
        :class:`DecoupledDegeneracyResult`.

    Notes:
        If ``H_tt`` or ``H_ww`` is singular (condition number > 1e12) no Schur
        complement exists; the result carries ``factorization_ok=False``, every
        axis flagged, zero-filled spectra and zero clamped eigenvalues.  No
        exception is raised — an under-constrained frame is an expected input,
        not a programming error.

    Raises:
        ValueError: if ``kappa_threshold`` or ``kappa_target`` is not positive.
    """
    H = np.asarray(H, dtype=float)
    assert H.shape == (6, 6), f"Expected 6×6 Hessian, got {H.shape}"

    if not kappa_threshold > 0.0:
        raise ValueError(f"kappa_threshold must be positive, got {kappa_threshold}")
    if kappa_target is None:
        kappa_target = kappa_threshold
    if not kappa_target > 0.0:
        raise ValueError(f"kappa_target must be positive, got {kappa_target}")

    # 1. Symmetrize and split into blocks (PCR order: translation first).
    H = 0.5 * (H + H.T)
    H_tt = H[:3, :3]
    H_tw = H[:3, 3:]
    H_wt = H[3:, :3]
    H_ww = H[3:, 3:]

    # 2. Full-matrix condition number — diagnostic only, see docstring.
    lam_full = np.linalg.eigvalsh(H)
    cond_full = _ratio(float(lam_full[-1]), float(lam_full[0]))

    # 3. Invertibility gate.  Without both diagonal blocks invertible there is
    #    no Schur complement to analyse.
    if not (_block_is_invertible(H_tt) and _block_is_invertible(H_ww)):
        zeros3 = np.zeros(3)
        zeros33 = np.zeros((3, 3))
        return DecoupledDegeneracyResult(
            S_t=zeros33.copy(),
            S_R=zeros33.copy(),
            eigenvalues_t=zeros3.copy(),
            eigenvalues_R=zeros3.copy(),
            eigenvectors_t=zeros33.copy(),
            eigenvectors_R=zeros33.copy(),
            aligned_lambda_t=zeros3.copy(),
            aligned_lambda_R=zeros3.copy(),
            aligned_basis_t=zeros33.copy(),
            aligned_basis_R=zeros33.copy(),
            contribution_t=zeros33.copy(),
            contribution_R=zeros33.copy(),
            degenerate_mask=np.ones(6, dtype=bool),
            clamped_lambda_t=zeros3.copy(),
            clamped_lambda_R=zeros3.copy(),
            cond_schur_t=float('inf'),
            cond_schur_R=float('inf'),
            cond_full=cond_full,
            factorization_ok=False,
            is_degenerate=True,
            num_constrained_dof=0,
        )

    # 4. Schur complements.  S_t: translation curvature after rotation has been
    #    optimally eliminated (and vice versa for S_R).
    S_t = H_tt - H_tw @ np.linalg.inv(H_ww) @ H_wt
    S_R = H_ww - H_wt @ np.linalg.inv(H_tt) @ H_tw
    S_t = 0.5 * (S_t + S_t.T)
    S_R = 0.5 * (S_R + S_R.T)

    # 5. Spectra (ascending).
    lam_t, V_t = np.linalg.eigh(S_t)
    lam_R, V_R = np.linalg.eigh(S_R)

    # 6. Per-axis relabelling.
    aligned_lambda_t, aligned_basis_t = _align_to_axes(lam_t, V_t)
    aligned_lambda_R, aligned_basis_R = _align_to_axes(lam_R, V_R)

    # 7. Per-axis, within-block degeneracy test (ratio vs ratio).
    lam_t_max = float(lam_t[-1])
    lam_R_max = float(lam_R[-1])
    mask_t = (lam_t_max / np.maximum(aligned_lambda_t, _LAMBDA_FLOOR)) > kappa_threshold
    mask_R = (lam_R_max / np.maximum(aligned_lambda_R, _LAMBDA_FLOOR)) > kappa_threshold
    degenerate_mask = np.concatenate([mask_t, mask_R])

    # 8. Clamping: raise the flagged axes to the target ratio, leave the rest.
    #
    #    MONOTONE BY CONSTRUCTION.  A flagged axis is lifted to the target level
    #    *or left where it is*, whichever is higher, so clamping can only ever
    #    raise an eigenvalue.  The per-axis maximum matters when
    #    kappa_target > kappa_threshold: an axis whose ratio falls between the
    #    two is flagged, but the target level sits *below* the eigenvalue it
    #    already has, and assigning it unconditionally would LOWER the spectrum.
    #    That would make the spectral deficit (clamped - aligned) negative and
    #    the regularizer built from it in dcreg_solve(mode='clamped') indefinite.
    #    With the maximum, the deficit is >= 0 elementwise for any positive
    #    kappa_target.
    #
    #    For kappa_target <= kappa_threshold (the default, since kappa_target
    #    defaults to kappa_threshold) the maximum is inert: a flagged axis has
    #    lam_max / aligned > kappa_threshold >= kappa_target by definition of the
    #    mask, hence aligned < lam_max / kappa_target and the target level always
    #    wins.  So this changes no result on the default path.
    clamped_lambda_t = aligned_lambda_t.copy()
    clamped_lambda_R = aligned_lambda_R.copy()
    level_t = max(lam_t_max / kappa_target, _CLAMP_FLOOR)
    level_R = max(lam_R_max / kappa_target, _CLAMP_FLOOR)
    clamped_lambda_t[mask_t] = np.maximum(aligned_lambda_t[mask_t], level_t)
    clamped_lambda_R[mask_R] = np.maximum(aligned_lambda_R[mask_R], level_R)

    return DecoupledDegeneracyResult(
        S_t=S_t,
        S_R=S_R,
        eigenvalues_t=lam_t,
        eigenvalues_R=lam_R,
        eigenvectors_t=V_t,
        eigenvectors_R=V_R,
        aligned_lambda_t=aligned_lambda_t,
        aligned_lambda_R=aligned_lambda_R,
        aligned_basis_t=aligned_basis_t,
        aligned_basis_R=aligned_basis_R,
        contribution_t=aligned_basis_t ** 2,
        contribution_R=aligned_basis_R ** 2,
        degenerate_mask=degenerate_mask,
        clamped_lambda_t=clamped_lambda_t,
        clamped_lambda_R=clamped_lambda_R,
        cond_schur_t=_ratio(lam_t_max, float(lam_t[0])),
        cond_schur_R=_ratio(lam_R_max, float(lam_R[0])),
        cond_full=cond_full,
        factorization_ok=True,
        is_degenerate=bool(np.any(degenerate_mask)),
        num_constrained_dof=int(6 - np.count_nonzero(degenerate_mask)),
    )


# ===========================================================================
# Decoupled mitigation solve
# ===========================================================================
#
# Independent implementation of the mitigation mathematics of
#
#     Hu et al., "DCReg: Decoupled Characterization for Efficient Degenerate
#     LiDAR Registration", IJRR 2026, arXiv:2509.06285.
#
# Written from the published equations only.

# Iteration floor for the PCG loop: a 6x6 SPD system's Krylov space is complete
# after 6 steps in exact arithmetic, so fewer than 6 can never be justified on
# convergence grounds.  Mirrors the published default behaviour.
_PCG_MIN_ITERATIONS = 6

# Below this, ``p^T A p`` is treated as a breakdown rather than a step length.
_PCG_CURVATURE_FLOOR = 1e-20


def _dcreg_preconditioner(deg: DecoupledDegeneracyResult) -> np.ndarray:
    """
    Assemble the block-diagonal DCReg preconditioner from a decoupled analysis.

    ``P = blkdiag(B_t, B_R)`` with ``B_t = A_t diag(1 / clamped_lambda_t) A_tᵀ``
    (and the mirror for rotation), where ``A_t`` is ``deg.aligned_basis_t``.

    Read that as ``P ≈ blkdiag(S_t⁻¹, S_R⁻¹)`` built on the *clamped* spectra:
    a well-conditioned direction is given its true inverse curvature, and a
    flagged one a bounded inverse instead of the enormous one its near-zero
    eigenvalue would produce.  That is the targeted part of the method — the
    weak directions are the only ones treated differently.

    Two structural consequences worth being explicit about:

    * ``P`` is block diagonal, so it deliberately carries no approximation of
      the translation/rotation coupling.  The Schur complements it is built
      from have already accounted for that coupling in their *spectra*, but the
      assembled operator does not reintroduce the off-diagonal blocks.
    * Because clamping only ever raises an eigenvalue and the floor
      ``_CLAMP_FLOOR`` is positive, ``P`` is symmetric positive definite
      whenever ``deg.factorization_ok`` — which is what a CG preconditioner has
      to be.

    Not meaningful for a ``factorization_ok=False`` result (the clamped spectra
    are zero-filled there); callers must gate on that first.
    """
    A_t = deg.aligned_basis_t
    A_R = deg.aligned_basis_R
    B_t = A_t @ np.diag(1.0 / np.maximum(deg.clamped_lambda_t, _CLAMP_FLOOR)) @ A_t.T
    B_R = A_R @ np.diag(1.0 / np.maximum(deg.clamped_lambda_R, _CLAMP_FLOOR)) @ A_R.T

    P = np.zeros((6, 6), dtype=float)
    P[:3, :3] = B_t
    P[3:, 3:] = B_R
    return P


def dcreg_solve(
    H: np.ndarray,
    g: np.ndarray,
    deg: DecoupledDegeneracyResult,
    mode: str = 'pcg',
    pcg_tolerance: float = 1e-6,
    pcg_max_iterations: int = 10,
) -> tuple:
    """
    Solve ``H dx = -g`` with DCReg-style targeted mitigation.

    Implements the mitigation mathematics of

        Hu et al., "DCReg: Decoupled Characterization for Efficient Degenerate
        LiDAR Registration", IJRR 2026, arXiv:2509.06285.

    ``H`` and ``g`` are in this library's DOF order ``[tx, ty, tz, ωx, ωy, ωz]``.

    Contract on ``deg``
    -------------------
    ``deg`` must come from :func:`analyse_hessian_decoupled` applied to the
    **raw** Hessian — the caller's responsibility, and the same rule
    :func:`apply_sr_solve` documents.  Analysing a damped ``H + λI`` would raise
    the small eigenvalues before the ratio test sees them and so under-report
    degeneracy, while the ``H`` passed *here* may perfectly well be the damped
    one (see :meth:`Registration.align`).  Analyse raw, solve damped.

    Modes
    -----
    **``mode='pcg'`` — the paper's published solver (§6).**

    A left-preconditioned Conjugate Gradient on ``A dx = -g`` with
    ``A = (H + Hᵀ)/2`` and the preconditioner of
    :func:`_dcreg_preconditioner`.

    BE CLEAR ABOUT WHAT THIS DOES AND DOES NOT DO.  The system being solved is
    the *original* one.  Preconditioning changes the trajectory of the
    iteration, never its fixed point, so **at full convergence this returns the
    plain Gauss-Newton step** — bit-comparable to ``np.linalg.solve(H, -g)``.
    The preconditioner's role, as the paper states it, is to improve the
    effective conditioning so CG converges in few iterations and the iterate is
    not swamped by round-off along the weak directions.  It does **not** modify
    the minimum.  Any regularizing effect comes only from stopping early: a
    truncated Krylov iterate is biased toward the well-conditioned directions,
    because CG resolves large-eigenvalue components first.

    How much conditioning it actually buys, stated precisely, because it is
    easy to overclaim.  Per block, the preconditioned spectrum is
    ``lambda_i / clamped_i``: exactly 1 on every unflagged axis, and
    ``lambda_i * kappa_target / lambda_max`` on a flagged one.  So

        ``cond(B_t S_t) = cond(S_t) / kappa_target``

    (and likewise for rotation) whenever at least one axis is unflagged.  The
    preconditioner *divides* the block condition number by ``kappa_target`` —
    it does not bound it by ``kappa_target``, nor by ``kappa_target**2``.
    Measured on the ``dcreg_minimal_example`` Hessian with
    ``kappa_target=10``: ``cond(S_t)`` 742.07 → 74.21, ``cond(S_R)`` 36.09 →
    3.61, and the full-system ``cond(P A)`` 1122.3 → 79.4.  A sufficiently
    degenerate block stays ill conditioned after preconditioning; the remedy is
    a larger ``kappa_target``, which is also a heavier distortion.

    On a 6-DOF registration problem the truncation effect is close to
    negligible in practice — the Krylov space is complete after 6 iterations, so
    the iteration runs to full convergence for essentially any tolerance one
    would actually ask for.  If you want the solve to move the answer, use
    ``mode='clamped'``.

    **``mode='clamped'`` — MAP-flavoured variant (paper Theorem 4, adapted).**

    Adds the per-block spectral deficit

        ``Gamma_t = A_t diag(clamped_lambda_t - aligned_lambda_t) A_tᵀ``

    (and the rotation mirror) to the corresponding diagonal block, then solves
    ``(A + blkdiag(Gamma_t, Gamma_R)) dx = -g``.  The deficit is exactly zero on
    every unflagged axis, so the regularization is Tikhonov-like but **only
    along the flagged aligned directions**, and identically zero elsewhere.
    With nothing flagged this reduces exactly to the Gauss-Newton solve.

    Unlike ``'pcg'``, this variant **does change the minimum** — that is the
    point of it.  It is also the aggressive one: an axis flagged by a tight
    ``kappa_threshold`` has its correction largely suppressed even if the data
    would in fact have recovered it.  Measured on a gently curved surface with
    ``cond(S_t) ~ 3e5`` and a genuine 0.6 m lateral offset, plain Gauss-Newton
    (and ``'pcg'``) recovered the offset to 6 significant figures while
    ``'clamped'`` at ``kappa_threshold=10`` left it essentially uncorrected.
    Choosing ``kappa_threshold`` is choosing how much recoverable signal you are
    willing to discard to bound the damage from unrecoverable directions.

    On ``kappa_target``.  ``Gamma`` is PSD for **any** positive
    ``kappa_target``, because :func:`analyse_hessian_decoupled` clamps
    monotonically: a flagged axis is assigned
    ``max(its own eigenvalue, lambda_max / kappa_target)``, so the deficit
    ``clamped - aligned`` is non-negative elementwise by construction.  A
    positive-definite ``H`` therefore yields a positive-definite ``H_reg``, and
    the ``np.linalg.solve`` below does not fall back.

    Raising ``kappa_target`` above ``kappa_threshold`` is the gentle setting: the
    target level drops below the eigenvalue of any flagged axis whose ratio sits
    between the two thresholds, so that axis keeps its own curvature and
    contributes nothing to ``Gamma``, while axes further out are still lifted.
    Lowering ``kappa_target`` below ``kappa_threshold`` regularizes harder.

    ADAPTATION NOTE (deviation from the paper).  The paper defines its MAP
    clamping on the Schur-*reduced* subproblems, i.e. on ``S_t`` and ``S_R``
    separately.  Adding the block-diagonal deficit to the full coupled ``H``
    regularizes the same directions by the same amounts, but does so on the
    coupled system rather than on the two reduced ones; the resulting step is
    not identical to solving the two reduced problems and recombining.  A
    consequence inherited from the D1 analysis: a weak direction that lives in
    the t↔R coupling can be flagged in *both* blocks and therefore regularized
    twice.  That is inherent to blockwise clamping, not a bug in this code.

    Fallback
    --------
    If ``deg.factorization_ok`` is ``False`` there are no Schur complements, no
    clamped spectra and no preconditioner, so no DCReg mitigation is defined.
    The solve degrades immediately to ``np.linalg.lstsq(H, -g)`` — a
    minimum-norm least-squares step, which for an exactly rank-deficient ``H``
    puts nothing along the null space.  ``used_fallback`` is ``True``.

    The same ``lstsq`` fallback catches a PCG run that exhausts its iterations
    or breaks down without meeting tolerance; the partial iterate is discarded
    rather than returned.  DEVIATION NOTE: the reference solver uses a
    column-pivoted QR at this point.  ``lstsq`` is the equivalent-in-semantics
    substitute here (both give a least-squares solution to a rank-deficient
    system, ``lstsq`` specifically the minimum-norm one); this is not a
    bit-parity reimplementation.

    Convergence test
    ----------------
    ``norm(r) <= pcg_tolerance * max(1.0, norm(b))`` — a hybrid absolute/relative
    criterion.  For a well-scaled problem (``norm(b) >= 1``) it is relative; for
    a nearly-converged one, where ``norm(b)`` has shrunk below 1, it becomes an
    absolute floor, which stops the tolerance from chasing an ever-smaller
    target as the ICP loop converges.

    Every iterate is checked for non-finiteness and the loop breaks to the
    fallback if any appears.

    Args:
        H: 6×6 Hessian in PCR DOF order.  May be damped; see the contract above.
        g: 6-element gradient in PCR DOF order.  The solve targets ``b = -g``.
        deg: :class:`DecoupledDegeneracyResult` from the **raw** Hessian.
        mode: ``'pcg'`` or ``'clamped'``.
        pcg_tolerance: Convergence tolerance for the ``'pcg'`` mode.
        pcg_max_iterations: Iteration budget for ``'pcg'``, floored at 6.

    Returns:
        ``(dx, info)`` where ``dx`` is the 6-element step in PCR DOF order and
        ``info`` is
        ``{'mode', 'pcg_iterations', 'pcg_converged', 'used_fallback',
        'relative_residual'}``.  The three PCG fields are ``None`` whenever the
        PCG iterate was not what got returned.

    Raises:
        ValueError: if ``mode`` is not ``'pcg'`` or ``'clamped'``; if
            ``pcg_tolerance`` is not positive and finite; or if
            ``pcg_max_iterations`` is not a non-negative integer.  The two PCG
            parameters are validated in both modes, so a nonsensical
            configuration is rejected rather than silently ignored by
            ``'clamped'``.
    """
    if mode not in ('pcg', 'clamped'):
        raise ValueError(
            f"Unknown dcreg mode {mode!r}; expected 'pcg' or 'clamped'"
        )
    if not (np.isfinite(pcg_tolerance) and pcg_tolerance > 0.0):
        raise ValueError(
            f"pcg_tolerance must be positive and finite, got {pcg_tolerance!r}"
        )
    if isinstance(pcg_max_iterations, bool) or not isinstance(
        pcg_max_iterations, (int, np.integer)
    ):
        raise ValueError(
            "pcg_max_iterations must be a non-negative integer, got "
            f"{pcg_max_iterations!r}"
        )
    if pcg_max_iterations < 0:
        raise ValueError(
            "pcg_max_iterations must be a non-negative integer, got "
            f"{pcg_max_iterations!r}"
        )

    H = np.asarray(H, dtype=float)
    assert H.shape == (6, 6), f"Expected 6×6 Hessian, got {H.shape}"
    b = -np.asarray(g, dtype=float)

    info = {
        'mode': mode,
        'pcg_iterations': None,
        'pcg_converged': None,
        'used_fallback': False,
        'relative_residual': None,
    }

    def _lstsq(matrix: np.ndarray) -> np.ndarray:
        info['used_fallback'] = True
        return np.linalg.lstsq(matrix, b, rcond=None)[0]

    # No Schur complements → no clamped spectra → nothing to mitigate with.
    if not deg.factorization_ok:
        return _lstsq(H), info

    A = 0.5 * (H + H.T)

    if mode == 'clamped':
        Gamma_t = (
            deg.aligned_basis_t
            @ np.diag(deg.clamped_lambda_t - deg.aligned_lambda_t)
            @ deg.aligned_basis_t.T
        )
        Gamma_R = (
            deg.aligned_basis_R
            @ np.diag(deg.clamped_lambda_R - deg.aligned_lambda_R)
            @ deg.aligned_basis_R.T
        )
        H_reg = A.copy()
        H_reg[:3, :3] += Gamma_t
        H_reg[3:, 3:] += Gamma_R
        try:
            return np.linalg.solve(H_reg, b), info
        except np.linalg.LinAlgError:
            return _lstsq(H_reg), info

    # ---- mode == 'pcg' ---------------------------------------------------
    P = _dcreg_preconditioner(deg)

    norm_b = float(np.linalg.norm(b))
    tol_abs = pcg_tolerance * max(1.0, norm_b)

    x = np.zeros(6, dtype=float)
    r = b.copy()
    z = P @ r
    p = z.copy()
    rz = float(r @ z)

    iterations = 0
    converged = float(np.linalg.norm(r)) <= tol_abs

    if not converged:
        for k in range(max(pcg_max_iterations, _PCG_MIN_ITERATIONS)):
            Ap = A @ p
            pAp = float(p @ Ap)
            if not np.isfinite(pAp) or abs(pAp) < _PCG_CURVATURE_FLOOR:
                break                       # breakdown / zero curvature

            alpha = rz / pAp
            if not np.isfinite(alpha):
                break

            x = x + alpha * p
            r = r - alpha * Ap
            iterations = k + 1

            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(r))):
                break

            if float(np.linalg.norm(r)) <= tol_abs:
                converged = True
                break

            z_new = P @ r
            rz_new = float(r @ z_new)
            if not np.isfinite(rz_new) or abs(rz) < _PCG_CURVATURE_FLOOR:
                break
            # Standard preconditioned-CG update: the denominator is the PREVIOUS
            # r·z, not the current residual against the previous z.  In exact
            # arithmetic PCG enforces r_{k+1}ᵀ z_k = 0, so using the latter would
            # divide by zero.
            beta = rz_new / rz
            p = z_new + beta * p
            rz = rz_new
            z = z_new

    if not converged:
        # Discard the partial iterate; it met no tolerance and may be garbage.
        return _lstsq(H), info

    info['pcg_iterations'] = iterations
    info['pcg_converged'] = True
    info['relative_residual'] = float(np.linalg.norm(r)) / max(1.0, norm_b)
    return x, info

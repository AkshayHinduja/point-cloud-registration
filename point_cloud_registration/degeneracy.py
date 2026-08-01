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

References:
    Zhang & Singh, "On Degeneracy of Optimization-based State Estimation
    Problems", IEEE International Conference on Robotics and Automation
    (ICRA), 2016 — J. Zhang, M. Kaess and S. Singh.

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
        ``aligned_lambda`` with the flagged axes raised to
        ``lambda_max_of_block / kappa_target``; unflagged axes are untouched.
        This is the spectrum a mitigation step (preconditioning / regularized
        solve) should use in place of the raw one.
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
           more); a value above it leaves the flagged axis still outside the
           threshold and is only useful if the caller wants a gentler nudge.

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
    clamped_lambda_t = aligned_lambda_t.copy()
    clamped_lambda_R = aligned_lambda_R.copy()
    clamped_lambda_t[mask_t] = max(lam_t_max / kappa_target, _CLAMP_FLOOR)
    clamped_lambda_R[mask_R] = max(lam_R_max / kappa_target, _CLAMP_FLOOR)

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

# V4 mechanisms, mathematical obligations and falsification

V4 is an independently namespaced replacement of the seven V3 proposed variants. The immutable release base is V3.0.1 (`3b1af9306a9ef9da541c5b0fabbe74dbafeb3011`). The runtime and reporting additions do not replace the legacy scientific implementations. A model change and a lossless execution change are different interventions; the numerical solver refinements below belong to the scientific configuration.

## Common mathematical types and data boundaries

A context C is a subset of TRAIN pairs. A target query is a TRAIN pair excluded from C, including all duplicates of that pair. No history feature, radius prior, covariance or decoder fit may use the excluded rating or timestamp. Validation is partitioned by unique pair without using Y into selection/shape-fit/shape-hold. All baselines use the same selection subset. TEST is absent from training and selection; tests replace its labels with NaNs.

Let rating support be [L,U], W=U−L, midpoint m and half-range h=W/2. A bounded context bias b(C,u,i) is shared conceptually across the proposed variants. Geometric residuals use a signed cosine c and nonnegative gain g≤1:

\[
\widehat y=b+gc\,[\mathbf1_{c\ge0}(U-b)+\mathbf1_{c<0}(b-L)].
\]

This guarantees support, not improved risk. Encoder outputs z are in R^K; direction q is on S^(K−1). Ridge precision is strictly positive. Context-specific uncertainty is a working conditional approximation, not a posterior over every network weight. Bias warmup, masked optimization steps, target counts, OOF costs and shape updates are reported separately. Equal K does not imply equal parameter count or equal compute.

## M1: decoder-informed tangent correction

Retain history-dependent radii, neural tangent steering, decoder reconstruction and the finite sphere exponential. With unit q, decoder rows D in R^(n×K), context residual t in R^n and radius r>0, define P=I−qqᵀ and

\[
H=r^2PD^TDP,\qquad a=rP(D^Tt-rD^TDq),\qquad v_C=(H+\lambda I)^{-1}a.
\]

The step minimizes the ridge-regularized **linearized** reconstruction error on the tangent subspace. V4 optionally information-filters the neural proposal, adds the weighted v_C correction, caps the tangent norm and applies Exp_q. The cavity step differentiates through its decoder, radius and context codes; the inherited information filter uses its declared detached curvature. There is no claim that a finite nonlinear Adam update minimizes the exact reconstruction objective.

Counterfactuals: no cavity, no information filter, no steering, fixed radii and no gradient surgery. The projection of conflicting auxiliary gradients is Euclidean; Adam momentum/decay remove a guarantee of descent for the final update. Failure signature: cavity overfits context noise or crowds out latent directions; test validation risk by degree and restrict cavity_strength using validation only.

## M2: OOF benefit with a context curvature penalty

Let p0=λ0 p_user+(1−λ0)p_item and d=p_user−p_item. A gated prediction p0+ηd has squared-error difference η²d²−2η(Y−p0)d. Therefore a conditional quadratic surrogate has minimizer

\[
\eta^*=\mathrm{clip}\left(\frac{B(X,Z)}{d^2+\alpha C(X)+\rho},-\lambda_0,1-\lambda_0\right),
\quad B(X,Z)=\mathbb E[(Y-p_0)d\mid X,Z],\quad C(X)=\mathbb E[d^2\mid X].
\]

X has eight context-only features, Z three pair expert features. Benefit uses X,Z; strictly positive softplus curvature heads see only X. An ensemble-based uncertainty threshold soft-shrinks B before gating. The αC term is a deliberate stabilizing penalty, **not** an identity equating observed d² to its conditional expectation. The coarse-risk control replaces denominator with C(X)+ρ and uses only X for benefit. Its ideal conditional-risk minimizer is exact only when the chosen sigma-algebra and conditional moments are correct.

Fit expert moments on genuine OOF TRAIN predictions grouped by unique pair. Fit router/early-stop on disjoint OOF subsets. Select λ0 and routing strength (including zero) on selection validation; never test. Smaller OOF contexts can shift the feature distribution at final refit, so no finite-sample certified risk improvement is asserted. Count-only gate, observed-curvature gate, legacy gate, coarse gate, fixed gate and consistency removal are distinct controls. Export OOF folds, features and targets so moment fitting can be audited.

## M3: context graph inference for log radii

The old persistent full-training delta is replaced, by default, with a center derived from the current context. The log-radius network gives bounded prior μ(C). A differentiable finite Gauss–Newton/CG step around μ solves a linearized ridge system on the context bipartite incidence graph. Bound it using t·tanh(step/t), add μ and clamp to the declared log-radius interval. Decoder/history codes also perturb directions before normalization. This ensures masked targets cannot re-enter through a stale all-TRAIN radius fit.

The graph covariance uses **the same context and center**, with fixed independent probe RNG and five-node sigmoid quadrature. It is a conditional local curvature approximation, not a guarantee of global posterior coverage. A single finite step need not reach MAP; its purpose is amortized correction. Controls retain the global radius block solver with its line-search objective diagnostic, remove context directions or remove graph uncertainty. Main failure modes are an overly strong prior, approximate covariance and context-to-full-history shift.

## M4: refine signed moments in prediction/dispersion coordinates

Keep separate positive and negative moment channels and the exact feasible polygon for bounded (including ordinal) responses. If P,N are partial moments about b, the point predictor is b+P−N. V4 obtains a context-decoder transfer

\[
\Delta=\frac W4\tanh\big[\tfrac12(z_u^Td_i+z_i^Te_u)\big],
\quad(P,N)_{proposal}\leftarrow(P+\Delta/2,N-\Delta/2).
\]

Project the proposal onto the existing polygon before prediction. Train P−N with point MSE and P+N with its own weighted dispersion loss toward |Y−b|. This separates prediction and dispersion coordinates; it does **not** claim the sample absolute residual is an unbiased target for a true conditional variance. Projection is piecewise differentiable and can create boundary optima.

Controls: no transfer, old partial-moment loss, one signed channel and no polygon. The no-polygon transfer control can predict outside [L,U]; its support certificate is widened to [L−W/4,U+W/4]. Reporting global OOB rate against [L,U] is still valid and must not be disguised by the wider support certificate. Full projected M4 uses the global support certificate, not a tight per-pair interval.

## M5: alternating ridge means and certified working-posterior solves

For one view, with initial encoder means Z0 and decoder prior D0, define

\[
F(Z,D)=\frac{\|Y_C-\langle Z,D\rangle_C\|_2^2}{2\sigma^2}
 +\frac\lambda2(\|Z-Z_0\|_F^2+\|D-D_0\|_F^2).
\]

Solve each decoder conditional ridge exactly. Then relax each code toward its conditional ridge solution with 0<ω≤1, **keeping the original Z0 prior**, and refit the decoder around D0. Each block update descends this fixed biconvex quadratic block objective when the other block is fixed; it does not guarantee descent in generalization RMSE or the later Gaussian-link objective. Default: one refinement, ω=.25.

Retain the joint linearized Gaussian precision, perturb-and-solve probes, user/decoder cross covariance and analytic Gaussian bilinear mean/variance. Add block-Cholesky preconditioning and check the true residual ||b−Ax||/||b||, tolerance 1e−4, maximum64 iterations, periodic checks every4. FP64 scalar accumulation is an explicit solver choice; vector dtype remains unchanged. Unconverged solves fail visibly. This corrects posterior computation; it need not improve RMSE. The original 12-step unpreconditioned solver is an explicitly labeled ablation, not the default valid posterior solver.

Controls: zero mean-refinement steps, legacy solver, block-independent posterior, zero cross covariance with retained joint marginals, no reconstruction. Independent inversion and post-hoc zeroing of cross covariance are mathematically different. Probe covariance is finite-rank; approximate variance is not automatically calibrated. Small signed mean cross correction is not evidence that absolute corrections vanish.

## M6: hierarchical angular reference with a soft residual moment prior

Retain monotone constrained angular transport, fixed-support geometry, exact spline mass/half-mass/first-moment constraints, zero-angle anchor, dual-view decoder coupling and slope budget. Estimate the angular reference using right-continuous histogram ECDFs at interior grid points. Shrink local baseline×gain×degree cells toward their degree parent, then global empirical reference, then isotropic reference using n/(n+τ). Empty cells borrow information rather than pretending a noisy tiny cell is a population law. Transport endpoint values are fixed at0 and1 by convention (including any atom exactly at the left endpoint); interior ties are included correctly.

Default V4 removes the old hard finite-catalog residual balancing. If every nonzero residual in a cell is positive, exact zero gain-weighted residual mean forces that whole positive signal to zero. Instead use a count-supported soft residual moment penalty; preserve hard balance as a control. The new default **does not** claim exact finite-catalog zero residual mean. The spline identities still concern the declared angular reference; hierarchical empirical references and finite catalogs do not establish rating calibration.

Controls: hard balance, old unshrunk reference, no moment prior, analytic transport, no transport, isotropic reference, degree-only reference, no radii and no decoder coupling. Report signal shrinkage and performance by cell support to distinguish reference estimation gains from merely loosening the hard constraint.

## M7: identity-aware mean ensemble with protected probability refinement

Keep independently initialized/bootstrapped bounded mean members and full mean-constrained ordinal shape heads. Add shared sparse identity-history encoders. Each member sees a fixed orthogonal rotation of their codes concatenated to the existing statistical features, preserving member diversity while making the means sensitive to which items/users appear in history, not only distributional summary statistics.

Train a convex combination of member MSE and ensemble-mean MSE; λ_ensemble=.25 by default. This is an explicit change in mean training, not a post-hoc mixture with a rival. Afterwards freeze every mean parameter and its cached state. Fit ordinal shape with NLL+.05RPS on probability-fit, early-stop on probability-hold. Hash protected parameters before/after. Export actual member means/PMFs and total/within/between variance. For continuous data use an explicitly endpoint-clipped same-mean beta mixture rather than quantizing ratings.

The ordinal exponential tilt enforces E[Y]=prescribed mean. Law of total variance is an identity for the declared mixture. Neither identity guarantees calibrated intervals, causal uncertainty or a posterior over all model weights. NLL in evaluation is pure likelihood, separate from the composite shape-training objective. Discrete equal-tail intervals can overcover even under a calibrated law; inspect cumulative reliability, randomized PIT, likelihood and sharpness together. Shape alone cannot repair a frozen biased mean. Controls: statistics-only mean input, one member, no ensemble-mean term, NLL-only shape, restricted shape and fixed radii.

## Execution policy and evidence obligations

Lossless saved-tensor hooks move exact values/dtypes to host/SSD; multi-backward support is tested. Failed CUDA candidates replay from initial model state and Python/NumPy/Torch CPU/selected-CUDA RNG, after releasing failed parameters and rolling back epoch logs. No precision, logical batch, masks, probes, step count or optimizer hyperparameter is adapted. Numerical overflow raises rather than silently changing the experiment. CPU parity tests do not prove CUDA bitwise parity or speedup.

Scientific evidence tiers: (1) algebraic identities and invariant tests; (2) one-split CPU validation pilots with test labels poisoned; (3) future matched-capacity/matched-target-budget ablations; (4) frozen multi-split locked test studies. Only tiers actually completed belong in results claims. Do not infer mechanism causality from defaults with different widths. Do not infer novelty from terminology: tangent refinement, OOF stacking, graph Gaussian approximation, signed partial moments, Laplace approximations and monotone transport each have prior art. The proposed contribution is their specific coupling and its falsifiable behavior, subject to a fuller literature review.


## V4.1 review extension

The release adds a context-density mixture objective, validation-selected common-protocol MBMF checkpoints, effective AutoRec Rprop rates, explicit MF/MBMF training budgets, and a mean-priority feasible-moment readout for M4. Weighted and Euclidean moment projections remain controls. The stronger simple mean-priority null outperformed the weighted projection in the one-split validation pilot; the metric is not claimed necessary or irreducible. See docs/V4_1_THEORY_FA.md and docs/V4_1_REVIEW_FA.md. No new real TEST result or SOTA/Q1 claim is made.

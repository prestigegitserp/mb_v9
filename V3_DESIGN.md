# V3 implementation and mathematical contracts

This is an executable research design, not an empirical claim of superiority or an assertion that its statistical building blocks are new. The seven families remain distinct. No old model is instantiated merely to leave its parameters unused. Original scientific implementations remain in `vendor_v8/`, `v9/` and `v2/`.

## Shared history and bounded predictions

Let the rating interval be $[L,U]$, width $W=U-L$ and half-width $h=W/2$. A shrinkage bias model supplies $b_{ui}\in(L,U)$ using train-only entity counts. Most geometric heads use

$$B(b,c,g)=b+gc\begin{cases}U-b&c\geq0\\b-L&c<0\end{cases},\qquad -1\leq c\leq1,\;0\leq g\leq1.$$

This gives global range bounds without clipping the final prediction. The continuous gate $\sigma(\log r_u+\log r_i-\log h+o)$ removes the exactly flat region of a hard radius-product cap; sigmoid saturation can still produce small gradients.

Every masked step removes all copies of query pairs from both histories. Statistical features, residual-amplitude proxies, histogram bins, timestamp normalization/extrema and uncertainty/reference calculations use that same context. No missing rating is represented as a genuine zero. Empty entities receive explicit population statistics. Residual amplitude is a shrinkage estimate of residual magnitude, **not** an identified confidence parameter.

Sparse scatter encoders pool value and observation-mask embeddings with degree normalization, then LayerNorm/GELU/linear projection. Hidden width is independent of K. Decoder Gram matrices accumulate in chunks; there is no required dense user-by-item rating matrix in a V3 model. The legacy dense data backend may still allocate its compatibility matrix.

All mean models minimize normalized squared error plus their declared penalties. M3 additionally uses its declared Gaussian working-noise precision. AdamW groups separate directions, biases and encoder matrices. Directions receive a geodesic-to-initialization penalty rather than Euclidean weight decay. Encoder decay is derived from the planned cumulative budget:

$$d=\frac{1-(1-\rho)^{1/T}}{\eta},\quad (1-\eta d)^T=1-\rho.$$

Actual optimizer steps and realized direct shrinkage are reported when early stopping truncates training. Code/decoder and bias priors are explicit loss terms. An epoch is five independent 20%-target masks by default; this is expected exposure, not an exact partition of the interactions.

## M1: identifiable tangent steering

The encoder supplies radius and a tangent proposal. With a unit population/ID direction $q$, projection $P=I-qq^T$ and selected decoder rows $D_j$:

$$H=r^2P\left(\sum_{j\in C_u}D_jD_j^T\right)P,\qquad v=(H+\lambda I)^{-1}Hv_0.$$

The proposal is reprojected into the tangent space and applied with a capped spherical exponential map. Zero-information directions retain the amortized/ID prior instead of inventing data-supported steering. H is a stopped working information matrix; the code does not differentiate through its construction. Reconstruction directly uses the same polar state, so its gradient can interfere with prediction.

A zero tangent limit returns the input direction with a zero tangent derivative, including a zero-information cold entity. The separate `no_steering` control removes **both** the history-conditioned prior-direction adjustment and the tangent proposal. Removing the information filter alone does not remove steering. Fixed-radius controls retain constant radii at half the radius cap for M1/M6/M7.

Shared encoder auxiliary gradients are capped relative to prediction gradients and, when conflicting, projected off the negative prediction component. The cap modifies only shared encoder coordinates; decoder reconstruction and nonshared priors retain their declared loss weights. A zero auxiliary-gradient budget makes the shared encoder gradient prediction-only. This is a Euclidean gradient-interference operation. Adam momentum, decay and finite step size prevent a theorem that the actual optimizer update always improves prediction loss. Ablations remove the filter and the surgery separately.

## M2: risk-based routing of distinct views

User and item views each have an independently supervised expert. Within-view consistency uses the more informative masked context as a stopped teacher for an additionally thinned context. There is no direct penalty forcing the two views to agree.

Three independently fitted fold experts produce genuine OOF training predictions. Every duplicate pair belongs to one fold. Given experts $p_1,p_2$, $d=p_1-p_2$ and fixed reference mixture $p_0=\lambda_0p_1+(1-\lambda_0)p_2$, the router learns

$$B(x)\approx E[(Y-p_0)d\mid x],\qquad
\eta(x)=\operatorname{clip}_{[-\lambda_0,1-\lambda_0]}
\frac{\operatorname{softthreshold}(\widehat B,\beta)}{d^2+\epsilon}.$$

This comes from the conditional squared-risk change $-2\eta B+\eta^2d^2$. Bootstrap router disagreement plus a model-based floor supplies $\beta$. The final gate is $\lambda_0+s\eta$, with $s\in\{0,.25,.5,1\}$ chosen on selection validation. The explicit zero-strength candidate retains the fixed mixture. This is validation protection, not a conditional risk guarantee. OOF experts use smaller histories than the final refit; this remaining distribution shift is recorded in checkpoint metadata. All OOF and router updates are budgeted and reported.

The count-only control directly fits $\eta(n_u,n_i)$ by OOF squared prediction risk. Its gate reads only the two log-count features, including at inference: it does not divide by the current pair's expert disagreement. The full risk router and this count-only control therefore have genuinely different information sets. `fixed_gate` retains the learned **global scalar** mixture; it is not an assertion that the scalar equals .5.

## M3: neural log-radius priors and graph uncertainty

Radii retain $r=r_0\exp\delta$. The encoder predicts a bounded prior center $\mu_\delta$; $\lambda\|\delta-\mu_\delta\|^2/2$ is a proper prior that fixes the bipartite scale gauge. Delta is excluded from Adam and updated by a trust-limited Gauss–Newton block solve.

With signless user-item incidence A and scalar normalized-score derivative j, the working precision is

$$H=A^T\operatorname{diag}(j^2/\sigma^2)A+\lambda I.$$

CG applies H without materializing the graph matrix. Perturb-and-solve probes use likelihood and proper-prior noise; in the exact fixed linear solve their covariance is $H^{-1}$. Pair log-radius variance is estimated from shared node draws, preserving covariance rather than adding marginal variances blindly. Five-node Gauss–Hermite quadrature integrates the gain.

Covariance is recomputed from the current masked context with fixed independent probe seeds. It uses the plug-in working derivative; finite CG/probes approximate the ideal conditional Gaussian and do not integrate encoder-weight uncertainty. The delta step differentiates the quadrature mean while holding the current covariance/history/prior fixed. Backtracking accepts only a nonincreasing **conditional block** objective; recomputing covariance afterward is not covered by a joint global-descent theorem. The same $\sigma^{-2}/2$ scaling is used in the mean training objective.

## M4: signed residual moments with exact feasible support

Raw amplitude priors are remapped as $p_0=r_0/(\tau+r_0)$ with a controlled interior median. Bounded learned shifts act on logit($p_0$), so both positive and negative radius channels are strictly within $\sqrt h$. A bias center is fitted during warmup and fixed thereafter.

Two angular channels predict $P\approx E[(Y-b)_+\mid x]$ and $N\approx E[(b-Y)_+\mid x]$, with mean $b+P-N$. Their population identity is exact for fixed b. Auxiliary squared losses supervise the individual partial moments as well as the final mean.

For discrete rating levels, the exact feasible set is the convex hull of $((r-b)_+,(b-r)_+)$ over support r. Its relevant vertices are $(0,b-L)$, $(U-b,0)$, $(r_+-b,0)$, $(0,b-r_-)$, where $r_-,r_+$ bracket b. For continuous labels the lower edge reaches the origin. Euclidean nearest-segment projection implements the polygon, with identity on its interior. Gradients exist almost everywhere; vertex/edge switching is nonsmooth. This gives compatible moments and bounded means, not guaranteed better point risk. Signed channels use K total coordinates split in two and require K>=4.

## M5: conditional decoder means and a joint working Gaussian

Each view fits its decoder mean using a differentiable ridge solve conditional on context encoder codes and a learned decoder prior. A matrix-free joint Jacobian then constructs a Gaussian working covariance over encoder **latent codes** and decoder factors, conditional on network weights. It is centered at the amortized/ridge state, not a fully optimized nonlinear joint posterior mode. Fixed seeds and context-specific solves make prediction reproducible after checkpoint restoration.

Let the covariance factors be $L_z,L_d$ and $C=\alpha L_zL_d^T$, $0\leq\alpha\leq1$. Shrinking the cross covariance while holding marginals fixed preserves positive semidefiniteness by convex combination of joint and independent covariance. A learned initial alpha of .05 limits spurious low-probe cross-correlation. For the constructed joint Gaussian,

$$E[z^Td]=\mu_z^T\mu_d+\operatorname{tr}C,$$

$$\operatorname{Var}(z^Td)=\mu_d^T\Sigma_z\mu_d+\mu_z^T\Sigma_d\mu_z
+2\mu_d^TC\mu_z+\operatorname{tr}(\Sigma_z\Sigma_d)+\operatorname{tr}(C^2).$$

The implementation computes these moments directly in covariance factors. Two distinct controls are provided. `zero_cross_covariance` sets alpha=0 **after** the joint solve, retaining its marginal covariance estimates. `independent_posterior` instead removes code/decoder cross blocks **before** inversion and uses exact Cholesky covariance factors for the resulting conditional entity blocks. The same plug-in means are retained to isolate covariance inference. This block-diagonal working Gaussian still does not integrate network weights or optimize a joint nonlinear MAP. The anchored probit score uses variance along the relevant bilinear direction. That probit head is a defined prediction rule; it is **not** the exact integral of a nonlinear/non-Gaussian posterior. Exported latent variances are not labelled calibrated rating variances. Probe count, CG tolerance/budget and prior strength require sensitivity analysis.

## M6: globally valid radii and constrained angular transport

Every encoder, decoder and free counterpart radius has the same cap $\sqrt h$, including both operands of convex radius coupling. Thus products divided by h stay in [0,1]. The temperature is dimension-aware; radii actively affect both reconstruction-linked views and their final predictions.

A train-context angular CDF combines the isotropic K-dimensional reference with an empirical catalogue-pair reference, separately for two views and cells of degree cohort, baseline and gain. Defaults use four degree cohorts, three baseline bins and three gain bins, giving 36 cells per view. This is binned conditioning, not exact conditioning on continuous covariates. Uniform catalogue IDs use no held-out labels. Small reference cells fall back to the isotropic CDF. References are rebuilt from each context and stored with that state's CDF and balance factors so one context cannot overwrite another state's prediction rule.

For $p=(b-L)/W$, a learned monotone quantile Q satisfies

$$Q(0)=0,\quad Q(1)=1,\quad Q(.5)=p,\quad \int_0^1Q(v)\,dv=p.$$

Cosine-spaced nodes include .5. Interval masses $\delta_j=\Delta Q_j$ are projected by a small convex QP onto positivity, upper-slope bounds and three linear equalities: total mass 1, left-half mass p, interval-midpoint moment $1-p$. Interpolation between anchor banks preserves every equality and monotonicity. Active-set differentiation projects gradients onto the free feasible tangent space; the projection is only piecewise differentiable. Entropy-interior initialization avoids the zero-gradient failure of a basic feasible LP vertex.

Outside the bank's [.02,.98] baseline range, a continuous convex closure combines the edge bank with an analytic finite-slope ramp law. For p<=.5 that law has left-half slope 2p and a terminal ramp of width $p/[2(1-p)]$; reflection supplies the other half. Its half-value and integral are both p. The closure mixing weight solves the target mean exactly and tends to 1 at the bank edge. Its finite slope bound is the weighted sum of its component bounds, rather than the interior bank's single cap. The analytic-power ablation intentionally uses the historical unconstrained endpoint-slope law.

The spline integral is a property for fixed p under a uniform transformed angle. It alone does **not** anchor the whole predictor when p or gain a varies with angle. For example, Q(v)=v, p=.5 and a=v give mean b+W/12, not b. The implemented head therefore adds the following explicit steps rather than claiming that independence.

First set $q=Q_p(F(c))$, $q_0=Q_p(F(0))$ and

$$s=\begin{cases}(q-q_0)/(1-q_0)&q\geq q_0\\(q-q_0)/q_0&q<q_0,\end{cases}\qquad R=s\begin{cases}U-b&s\geq0\\b-L&s<0.\end{cases}$$

Denominators have explicit numerical floors. Thus R is bounded, monotone in c for fixed baseline/reference, and R(0)=0 even for an asymmetric empirical CDF. For each view/cell on the sampled context reference, compute $P=\sum a_j(R_j)_+$ and $N=\sum a_j(-R_j)_+$. Outside a shared numerical floor, take $t_+=\min(1,N/P)$ and $t_-=\min(1,P/N)$, with identity for an empty/zero side where its ratio is undefined. Both scales stay in [0,1]. Below the shared floor both may shrink; the equality $Pt_+=Nt_-$ still holds. Predict

$$\mu=b+a\{t_+(R)_+-t_-(-R)_+\}.$$

This preserves global bounds and the zero-angle anchor, and enforces $\sum(\mu_j-b_j)=0$ on the actual finite reference **including correlated p, gain and angle**. The factors are differentiable almost everywhere through P and N; CDF estimation and bin membership are stopped working quantities. A one-sided nonzero reference must contract that side to zero to satisfy all these constraints. Empty cells have identity scales, and no unseen-cell calibration guarantee is asserted. The identity concerns the sampled catalogue reference, not observed ratings or unknown user-specific calibration. Binned conditioning, small cells, shrinkage and sampling error require sensitivity analysis. Ablations separately remove balancing, continuous-covariate bins, learned spline, radii and direct decoder coupling.

## M7: protected means and a full ordinal law

Three mean members use bootstrap-weighted squared error, independent parameters, feature dropout and count-dependent support. Each member's radius multiplies its directional residual, so it is not an unused decoration. K is per member; total capacity exceeds a single K-dimensional model.

After selecting mean checkpoints, shape heads see only probability-fit validation and stop on a separate probability-hold role. All inputs and mean values passed into shape fitting are detached; optimizer groups contain only shape-head parameters. A hash of all protected mean parameters/buffers is checked before and after fitting.

For ordinal support $r_1,...,r_J$ and fixed member mean m, arbitrary logits a are exponentially tilted:

$$\pi_j=\frac{\exp(a_j+\lambda r_j)}{\sum_l\exp(a_l+\lambda r_l)},\qquad\sum_jr_j\pi_j=m.$$

A double-precision scalar root solve determines lambda. The forward solve removes the affine-logit gauge and brackets the root using the largest adjacent logit slope; it verifies convergence instead of silently violating the mean for large logits. The implicit shape Jacobian is $\operatorname{diag}\pi-\pi\pi^T-vv^T/\operatorname{Var}_\pi(r)$, $v_j=\pi_j(r_j-m)$. Its generic rank J-2 retains the full mean-constrained family, rather than a single temperature. The two affine-logit gauge directions are removed by this derivative. Numerical endpoint/floor tolerances remain explicit in code.

The ensemble mixes member probabilities and retains the ensemble point mean. Total variance is exactly within-member variance plus variance of member means. Metrics include actual ordinal NLL, multiclass Brier, RPS and discrete interval coverage; no Gaussian NLL is fabricated for an ordinal PMF.

Continuous train labels are detected even when a dataset manifest declares a nominal step. That path uses two beta components with the **same member mean**, learned concentrations and mixture weight. Log mixture weights are computed directly as log-sigmoid(logit) and log-sigmoid(-logit); taking logs of a rounded sigmoid is avoided so saturated logits retain finite gradients. A fixed inward affine map of size 1e-5 is part of the trained point head, ensuring an interior mean even if its bias sigmoid saturates numerically; probability fitting does not subsequently move that mean. Density NLL clips normalized endpoint labels by 1e-5 and is labelled accordingly. This is a distinct parametric law, not the full ordinal family; continuous and discrete NLL values must not be compared as the same likelihood. Shape and router fitting use minibatches and log their extra optimizer budgets. The restricted-shape ablation uses quadratic ordinal logits or a single beta law on continuous datasets.

## Experiment and recovery contracts

Original train/test are unchanged. Original validation is partitioned by unique pair and seed without reading ratings, identically for baselines and replacements: selection .60, probability fit .20, probability stopping .20. Consequently historical full-validation V2 metrics are not the same validation population. Fair architectural claims require a matched rerun and multiple splits/seeds.

Scoped runtime hooks are restored even after exceptions. Baseline constructors, trainers and checkpoint formats are delegated to the preserved implementation. Config fingerprints include nested V3 settings. V3 checkpoint signatures include train data/timestamps and all validation roles, while intentionally excluding test labels. Strict restoration rejects incompatible data, versions and state shapes.

Search changes actual V3 settings and removes obsolete angle/old-loss sweeps. Ablations disable implemented mechanisms. Probability-NLL model selection is a secondary study with its own label. Before a locked refit, archived scan settings, seeds, implementation identity, selected config key and train/validation data signatures must match. Final retention flags are restored after scans so `all` retains its selected models and predictions. No test statistic enters selection, OOF, shape fitting or stopping. CPU test coverage checks these contracts; GPU parity and publication-grade improvement remain unverified.

V3.0.1 changes the source identity, configuration and M6 forward rule. Strict restoration intentionally rejects old V3 checkpoints; use a fresh output directory and retrain. Training records the actual last epoch, whether patience or the epoch budget ended fitting, and whether the best checkpoint lies at the budget boundary. Reaching that boundary is not evidence of convergence.

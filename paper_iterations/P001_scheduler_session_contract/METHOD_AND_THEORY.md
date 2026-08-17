# P001 Method and Theory: Registered Diffusion States and Auditable CL Estimands

## 1. Scheduler-consistent student-support distillation

Let (x_0) denote a normalized trajectory and let the forward diffusion state be

\[
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\]

A distillation query is well-defined only if the tensor supplied to a denoiser has the same registered
time (t) as its time embedding. A DDIM transition from (t_k) must additionally land on the next
registered query time (t_{k+1}). With a 1,000-step training grid and the truncated query sequence
((10,0)), the scheduler must therefore use stride 10, equivalently 100 inference steps.

The historical modified-RAP inference heuristic instead constructed (x_8), queried it with the
(t=10) embedding, configured 1,000 inference steps, and then treated the scheduler's (10\to9)
transition as the state for the next (t=0) query. This creates two separate errors:

\[
x_8 \not\equiv x_{10}, \qquad
\Phi^{\Delta=1}_{10}(x_8) \in \mathcal X_9 \not\equiv \mathcal X_0.
\]

P001 registers the sequence

\[
x_{10}^{S}\xrightarrow{f_S(\cdot,10)}
\Phi^{\Delta=10}_{10\rightarrow0}(x_{10}^{S})=x_0^{S},
\]

and performs no unused scheduler transition after the final (t=0) query. The planning loss remains

\[
\mathcal L_{\mathrm{plan}}
=\frac1{|\mathcal T|}\sum_{t\in\{10,0\}}
\left\|f_S(x_t,t,c_S)-\operatorname{sg}f_T(x_t,t,c_T)\right\|_2^2
+\beta\,D_{\mathrm{KL}}\!\left(p_T(\cdot|x_t,t,c_T)\|p_S(\cdot|x_t,t,c_S)\right).
\]

OPD constructs (x_t) by advancing the student's generated state; LwF constructs (x_t) by adding
matched noise to the ground-truth trajectory. All other query code, timesteps, teacher, response loss,
mode KL, microbatch, and forward counts are shared. A negative-control unit test supplies the same
registered states to both paths and verifies identical scalar losses and parameter gradients.

## 2. The update is explicitly a semi-gradient

The transition-driving student response is detached. The implemented update is therefore

\[
g_{\mathrm{OPD}}(\theta)=
\mathbb E_{\epsilon}\left[
\partial_\theta\ell\bigl(\theta,x_t^S(\bar\theta;\epsilon)\bigr)
\right],
\]

where (ar\theta) denotes the stop-gradient copy used to construct the sampled state. It omits the
pathwise term ((\partial_x\ell)(\partial_\theta x_t^S)). This controls memory and avoids a second-order
rollout graph, but it is not automatically the gradient of a global state-dependent objective.
Czarnecki et al. show an analogous subtlety for student-policy/on-policy distillation: useful updates
need not form a conservative gradient field. Consequently P001 claims only a correctly registered
semi-gradient estimator; convergence or NAVSIM benefit must be established empirically in P002.

## 3. EWC importance proxy

For the official composite driving loss (ell_n), the diagonal importance statistic is now

\[
D_i=\frac1N\sum_{n=1}^{N}\left(\partial_{\theta_i}\ell_n\right)^2.
\]

The previous batch implementation squared an aggregate gradient,
((\sum_n g_{n,i})^2), which contains cross-example terms and is not the per-example second moment.
P001 enforces one-example backward calls and labels (D_i) a loss-gradient second-moment proxy, not
the Fisher information of a normalized probabilistic driving model. This distinction follows the
limitations identified by Kunstner et al. for empirical-Fisher substitutions.

## 4. A-GEM buffer-state intervention

Given current gradient (g) and replay reference gradient (g_r), A-GEM applies

\[
\tilde g=
\begin{cases}
g-\dfrac{g^\top g_r}{g_r^\top g_r}g_r,&g^\top g_r<0,\\
g,&\text{otherwise}.
\end{cases}
\]

The replay forward must provide gradients but must not constitute a second update to BatchNorm running
statistics. P001 snapshots every registered model buffer immediately after the current-data forward,
runs the replay forward/backward in train mode, and restores all buffers before the optimizer step.
This keeps the response function matched while making the auxiliary pass state-neutral.

## 5. Session-level estimand

Segmented NAVSIM folders from one timestamp–vehicle capture are dependent pieces of one deployment
session. For method (B) relative to baseline (A), P001 defines

\[
d_s^{(m)}=\frac{1}{|I_s|}\sum_{i\in I_s}
\left(y^{(m)}_{B,i}-y^{(m)}_{A,i}\right),
\qquad
\widehat\Delta^{(m)}=\frac1{|S|}\sum_{s\in S}d_s^{(m)}.
\]

Confidence intervals resample complete sessions. Three-seed results use a hierarchical bootstrap that
resamples training seeds and then sessions within each sampled seed. Invalid or missing PDM rows fail
closed. Families of metric comparisons use Holm correction. This changes the inferential unit, not
the underlying NAVSIM metric.

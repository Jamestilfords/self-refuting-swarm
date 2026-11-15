import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "A Dirichlet's marginal predictive variances are allocated across intents differently from its raw evidence or vacuity. "
    "Train-fitted, predicted-intent-specific deviations in this uncertainty-allocation spectrum flag atypical evidential competition shapes."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0050", "m_0092"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def features(alpha):
        a = np.clip(np.asarray(alpha, dtype=float), 1e-8, 1e12)
        total = np.sum(a, axis=1)
        p = a / total[:, None]

        # Apart from a common factor 1 / (total + 1), Dirichlet marginal
        # variances are p_k * (1 - p_k).  Normalize them to isolate how
        # uncertainty is distributed over competing intents.
        marginal_var = p * (1.0 - p)
        weights = marginal_var / np.maximum(
            np.sum(marginal_var, axis=1, keepdims=True), 1e-12
        )

        powers = np.array([0.5, 1.5, 2.0, 3.0, 4.0])
        spectrum = np.stack(
            [np.log(np.maximum(np.sum(weights ** r, axis=1), 1e-12)) for r in powers],
            axis=1,
        )
        evidence = np.log1p(total)[:, None]
        return np.concatenate([spectrum, evidence], axis=1)

    xtr = features(ctx.h.alpha(tr.logits))
    xev = features(ctx.h.alpha(ev.logits))
    ytr = np.argmax(tr.logits, axis=1)
    yev = np.argmax(ev.logits, axis=1)

    # The final coordinate is log total evidence.  Learn the expected
    # uncertainty-spectrum shape as a smooth quadratic function of it.
    def design(x):
        s = x[:, -1]
        return np.column_stack([np.ones_like(s), s, s * s])

    def fit_group(x):
        target = x[:, :-1]
        d = design(x)
        ridge = 1e-3 * np.eye(d.shape[1])
        beta = np.linalg.solve(d.T @ d + ridge, d.T @ target)
        residual = target - d @ beta

        if residual.shape[0] > 1:
            cov = (residual.T @ residual) / max(residual.shape[0] - 1, 1)
        else:
            cov = np.eye(target.shape[1])
        scale = float(np.trace(cov) / cov.shape[0])
        cov = cov + (1e-4 * max(scale, 1e-6)) * np.eye(cov.shape[0])
        precision = np.linalg.pinv(cov)
        return beta, precision

    global_model = fit_group(xtr)
    models = {}
    for c in np.unique(ytr):
        idx = ytr == c
        if np.sum(idx) >= 12:
            models[int(c)] = fit_group(xtr[idx])

    out = np.empty(xev.shape[0], dtype=float)
    for c in np.unique(yev):
        idx = yev == c
        beta, precision = models.get(int(c), global_model)
        residual = xev[idx, :-1] - design(xev[idx]) @ beta
        out[idx] = np.einsum("ij,jk,ik->i", residual, precision, residual)

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)

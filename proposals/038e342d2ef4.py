import numpy as np
from scipy.special import logsumexp

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "Centered tempered free-energy curvature captures the full winner-versus-rival logit spectrum while removing arbitrary per-item logit offsets. "
    "Class-conditioned surprise in this thermodynamic shape can flag OOS competition patterns beyond ordinary energy or maximum probability."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0074"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    temps = np.array([0.35, 0.60, 1.70, 3.50], dtype=float)

    def curvature_features(logits):
        base = logsumexp(logits, axis=1)
        free = np.column_stack(
            [t * logsumexp(logits / t, axis=1) for t in temps]
        )
        delta = free - base[:, None]

        # Remove the best linear temperature trend, retaining curvature of the
        # tempered free-energy path rather than its overall magnitude.
        x = temps - temps.mean()
        denom = np.sum(x * x)
        intercept = delta.mean(axis=1, keepdims=True)
        slope = ((delta - intercept) @ x / denom)[:, None]
        return delta - intercept - slope * x[None, :]

    xtr = curvature_features(tr.logits)
    xev = curvature_features(ev.logits)
    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]
    d = xtr.shape[1]

    global_mean = xtr.mean(axis=0)
    global_var = xtr.var(axis=0) + 1e-8
    means = np.empty((n_classes, d), dtype=float)
    variances = np.empty((n_classes, d), dtype=float)

    for c in range(n_classes):
        xc = xtr[pred_tr == c]
        n = xc.shape[0]
        if n == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = xc.mean(axis=0)
            class_var = xc.var(axis=0)
            shrink = n / (n + 12.0)
            variances[c] = shrink * class_var + (1.0 - shrink) * global_var + 1e-8

    z = (xev - means[pred_ev]) / np.sqrt(variances[pred_ev])
    return np.log1p(np.sum(z * z, axis=1))

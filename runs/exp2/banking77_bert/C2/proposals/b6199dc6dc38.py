import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Trailing within-class covariance directions retain subtle residual variation that full-space metrics can dilute. Combining this residual with train-calibrated logit uncertainty prioritizes inputs that are both off-manifold in low-variance directions and weakly supported by the classifier."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    r_train = np.asarray(partial.bottom(tr.pooled), dtype=float)
    r_eval = np.asarray(partial.bottom(ev.pooled), dtype=float)

    u_train = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    u_eval = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    r_med = np.median(r_train)
    r_scale = np.median(np.abs(r_train - r_med)) * 1.4826
    u_med = np.median(u_train)
    u_scale = np.median(np.abs(u_train - u_med)) * 1.4826

    r_scale = max(float(r_scale), 1e-8)
    u_scale = max(float(u_scale), 1e-8)

    residual = np.maximum((r_eval - r_med) / r_scale, -8.0)
    uncertainty = np.clip((u_eval - u_med) / u_scale, -8.0, 8.0)

    scores = residual + 0.35 * np.tanh(uncertainty)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)

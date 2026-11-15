import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "OOS inputs can induce Dirichlet concentration profiles that are atypical for the classifier-routed intent, including unusual combinations of total evidence, concentration entropy, and winner separation. A class-conditional profile distance uses the shape of the Dirichlet parameters rather than a score monotone in energy or vacuity."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0103"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def profile(logits):
        alpha = np.asarray(ctx.h.alpha(logits), dtype=float)
        alpha = np.maximum(alpha, 1e-12)
        total = np.sum(alpha, axis=1)
        p = alpha / total[:, None]
        entropy = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=1)
        entropy /= np.log(max(p.shape[1], 2))
        top = np.max(p, axis=1)
        second = np.partition(p, -2, axis=1)[:, -2] if p.shape[1] > 1 else np.zeros_like(top)
        gap = top - second
        return np.column_stack((np.log1p(total), entropy, gap))

    z_tr = profile(tr.logits)
    z_ev = profile(ev.logits)
    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]

    global_mean = np.mean(z_tr, axis=0)
    global_var = np.var(z_tr, axis=0) + 1e-6
    means = np.tile(global_mean, (k, 1))
    variances = np.tile(global_var, (k, 1))

    for c in range(k):
        mask = pred_tr == c
        if np.sum(mask) >= 4:
            means[c] = np.mean(z_tr[mask], axis=0)
            variances[c] = np.var(z_tr[mask], axis=0) + 1e-6

    delta = z_ev - means[pred_ev]
    scores = np.sum((delta * delta) / variances[pred_ev], axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)

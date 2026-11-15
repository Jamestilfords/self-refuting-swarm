import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "Out-of-scope inputs can have Dirichlet concentration shapes unlike those observed for any in-scope intent, even when their total evidence resembles an in-scope example. This uses class-conditional robust deviations in both total concentration and normalized concentration entropy, so it is not a monotone transform of energy or vacuity."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-12)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-12)
    k = a_tr.shape[1]
    logk = np.log(max(k, 2))

    def shape_features(alpha):
        total = np.sum(alpha, axis=1)
        p = alpha / total[:, None]
        concentration_entropy = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=1) / logk
        return np.column_stack((np.log(total), concentration_entropy))

    z_tr = shape_features(a_tr)
    z_ev = shape_features(a_ev)
    classes = np.unique(tr.labels)

    best = np.full(z_ev.shape[0], np.inf, dtype=float)
    for c in classes:
        zc = z_tr[tr.labels == c]
        center = np.median(zc, axis=0)
        mad = np.median(np.abs(zc - center), axis=0)
        scale = np.maximum(1.4826 * mad, 1e-4)
        deviation = np.abs((z_ev - center) / scale)
        class_score = deviation[:, 0] + deviation[:, 1]
        best = np.minimum(best, class_score)

    return np.nan_to_num(best, nan=0.0, posinf=1e6, neginf=0.0)

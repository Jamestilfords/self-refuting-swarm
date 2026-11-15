import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "OOS inputs can have Dirichlet concentration compositions and total evidence unlike the training examples of their predicted intent. This class-conditional evidential-shape deviation uses both relative concentration geometry and evidence strength, so it is not a monotone transform of energy or vacuity."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    def evidential_features(logits):
        alpha = np.asarray(ctx.h.alpha(logits), dtype=float)
        alpha = np.nan_to_num(alpha, nan=1.0, posinf=1e8, neginf=eps)
        alpha = np.clip(alpha, eps, 1e8)
        total = np.sum(alpha, axis=1, keepdims=True)
        log_comp = np.log(alpha / np.maximum(total, eps))
        log_comp = log_comp - np.mean(log_comp, axis=1, keepdims=True)
        strength = np.log(np.maximum(total, eps))
        return np.concatenate([log_comp, strength], axis=1)

    xtr = evidential_features(tr.logits)
    xev = evidential_features(ev.logits)
    labels = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = xtr.shape[1]

    global_mean = np.mean(xtr, axis=0)
    global_var = np.var(xtr, axis=0) + 1e-4
    means = np.tile(global_mean, (k, 1))
    variances = np.tile(global_var, (k, 1))

    for c in range(k):
        xc = xtr[labels == c]
        if xc.shape[0] >= 2:
            means[c] = np.mean(xc, axis=0)
            local_var = np.var(xc, axis=0)
            shrink = xc.shape[0] / (xc.shape[0] + 10.0)
            variances[c] = shrink * local_var + (1.0 - shrink) * global_var + 1e-4
        elif xc.shape[0] == 1:
            means[c] = xc[0]
            variances[c] = global_var + 1e-4

    pred = np.argmax(ev.logits, axis=1)
    diff = xev - means[pred]
    z2 = np.sum((diff * diff) / variances[pred], axis=1)

    train_pred = np.argmax(tr.logits, axis=1)
    tr_diff = xtr - means[train_pred]
    tr_z2 = np.sum((tr_diff * tr_diff) / variances[train_pred], axis=1)
    center = np.median(tr_z2)
    scale = np.median(np.abs(tr_z2 - center)) + 1e-6

    return np.nan_to_num((z2 - center) / scale, nan=0.0, posinf=1e6, neginf=-1e6)

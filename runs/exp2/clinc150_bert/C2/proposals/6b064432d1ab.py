import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "In-scope examples should exhibit class-specific Dirichlet evidence allocation patterns, while OOS inputs can have atypical normalized concentration shapes even when their total evidence or energy appears confident. This score measures diagonal Mahalanobis novelty of the log Dirichlet composition relative to the predicted class."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    a_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), eps)

    z_tr = np.log(a_tr) - np.log(np.sum(a_tr, axis=1, keepdims=True))
    z_ev = np.log(a_ev) - np.log(np.sum(a_ev, axis=1, keepdims=True))

    pred_tr = np.argmax(a_tr, axis=1)
    pred_ev = np.argmax(a_ev, axis=1)
    k = a_tr.shape[1]

    global_mean = np.mean(z_tr, axis=0)
    global_var = np.var(z_tr, axis=0) + 1e-4
    means = np.tile(global_mean, (k, 1))
    variances = np.tile(global_var, (k, 1))

    for c in range(k):
        mask = pred_tr == c
        if np.sum(mask) >= 4:
            class_z = z_tr[mask]
            means[c] = np.mean(class_z, axis=0)
            variances[c] = np.var(class_z, axis=0) + 1e-4

    diff = z_ev - means[pred_ev]
    score_values = np.mean((diff * diff) / variances[pred_ev], axis=1)
    return np.maximum(score_values, 0.0) + 1e-12

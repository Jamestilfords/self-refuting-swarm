import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "Dirichlet concentration shape, after removing total evidence by a centered log-ratio transform, "
    "captures which intents receive relative evidential support. OOS queries can have atypical evidence-allocation profiles even when their total concentration is not low."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0097"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    z_tr = np.log(a_tr)
    z_tr = z_tr - z_tr.mean(axis=1, keepdims=True)
    z_ev = np.log(a_ev)
    z_ev = z_ev - z_ev.mean(axis=1, keepdims=True)

    labels = tr.labels.astype(int)
    pred_ev = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]
    dim = z_tr.shape[1]

    global_mean = z_tr.mean(axis=0)
    global_var = z_tr.var(axis=0) + 1e-5
    means = np.tile(global_mean, (n_classes, 1))
    vars_ = np.tile(global_var, (n_classes, 1))

    for c in range(n_classes):
        members = z_tr[labels == c]
        if members.shape[0] >= 2:
            means[c] = members.mean(axis=0)
            class_var = members.var(axis=0)
            shrink = min(0.8, dim / (members.shape[0] + dim))
            vars_[c] = (1.0 - shrink) * class_var + shrink * global_var + 1e-5

    delta = z_ev - means[pred_ev]
    score_values = np.mean((delta * delta) / vars_[pred_ev], axis=1)
    return np.nan_to_num(score_values, nan=0.0, posinf=1e12, neginf=0.0)

import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The centered log Dirichlet concentrations describe how evidential mass is distributed across intents independently of total evidence. "
    "OOS queries can have an atypical concentration shape for their logit-predicted intent even when vacuity or energy appears in-distribution."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0011"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    alpha_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-12)
    alpha_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-12)

    # Centered log-ratios remove the total concentration scale, retaining
    # only the relative evidential allocation across classes.
    shape_tr = np.log(alpha_tr)
    shape_tr = shape_tr - shape_tr.mean(axis=1, keepdims=True)
    shape_ev = np.log(alpha_ev)
    shape_ev = shape_ev - shape_ev.mean(axis=1, keepdims=True)

    labels = tr.labels.astype(int)
    n_classes = tr.logits.shape[1]
    dim = shape_tr.shape[1]

    global_mean = shape_tr.mean(axis=0)
    global_var = shape_tr.var(axis=0) + 1e-5

    means = np.empty((n_classes, dim), dtype=float)
    variances = np.empty((n_classes, dim), dtype=float)

    for c in range(n_classes):
        members = shape_tr[labels == c]
        count = members.shape[0]
        if count == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = members.mean(axis=0)
            class_var = members.var(axis=0)
            # Shrink sparse class estimates toward the global shape variance.
            weight = count / (count + 10.0)
            variances[c] = weight * class_var + (1.0 - weight) * global_var + 1e-5

    predicted = np.argmax(ev.logits, axis=1)
    delta = shape_ev - means[predicted]
    return np.mean((delta * delta) / variances[predicted], axis=1)

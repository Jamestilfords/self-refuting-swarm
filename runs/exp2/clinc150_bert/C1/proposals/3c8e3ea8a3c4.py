import numpy as np

FAMILY = "logit"
ARM = "edl"
RATIONALE = "OOS inputs can induce Dirichlet concentration patterns whose normalized evidence allocation is atypical for the class they appear to predict. A class-conditional diagonal surprise score over log Dirichlet proportions detects these shape deviations without reducing to total evidence or energy."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.clip(ctx.h.alpha(tr.logits), 1e-8, 1e12)
    a_ev = np.clip(ctx.h.alpha(ev.logits), 1e-8, 1e12)

    z_tr = np.log(a_tr) - np.log(np.sum(a_tr, axis=1, keepdims=True))
    z_ev = np.log(a_ev) - np.log(np.sum(a_ev, axis=1, keepdims=True))

    labels = np.asarray(tr.labels, dtype=int)
    n_classes = a_tr.shape[1]
    global_mean = np.mean(z_tr, axis=0)

    means = np.empty((n_classes, z_tr.shape[1]), dtype=float)
    for c in range(n_classes):
        members = z_tr[labels == c]
        means[c] = np.mean(members, axis=0) if len(members) else global_mean

    centered = z_tr - means[np.clip(labels, 0, n_classes - 1)]
    var = np.mean(centered * centered, axis=0)
    var = np.maximum(var, 1e-6)

    pred = np.argmax(a_ev, axis=1)
    delta = z_ev - means[pred]
    return np.sum((delta * delta) / var, axis=1)

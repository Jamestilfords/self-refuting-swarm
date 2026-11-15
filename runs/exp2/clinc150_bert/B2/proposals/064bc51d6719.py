import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "Centered log-ratios of Dirichlet concentrations isolate the relative shape of evidential support "
    "across intents, removing total concentration (and thus vacuity) while flagging atypical evidence patterns."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    alpha_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    alpha_ev = np.maximum(ctx.h.alpha(ev.logits), eps)

    # Centered log-ratio representation removes the concentration scale and
    # preserves only the Dirichlet evidence allocation shape.
    clr_tr = np.log(alpha_tr)
    clr_tr = clr_tr - clr_tr.mean(axis=1, keepdims=True)
    clr_ev = np.log(alpha_ev)
    clr_ev = clr_ev - clr_ev.mean(axis=1, keepdims=True)

    labels = tr.labels.astype(int)
    n_classes = tr.logits.shape[1]
    dim = clr_tr.shape[1]

    counts = np.bincount(labels, minlength=n_classes).astype(float)
    global_mean = clr_tr.mean(axis=0)
    global_var = clr_tr.var(axis=0) + 1e-4

    sums = np.zeros((n_classes, dim), dtype=float)
    np.add.at(sums, labels, clr_tr)
    means = sums / np.maximum(counts[:, None], 1.0)

    # Shrink sparse class estimates toward the global evidence-shape profile.
    shrink = counts / (counts + 12.0)
    means = shrink[:, None] * means + (1.0 - shrink[:, None]) * global_mean

    centered = clr_tr - means[labels]
    sq_sums = np.zeros((n_classes, dim), dtype=float)
    np.add.at(sq_sums, labels, centered * centered)
    variances = sq_sums / np.maximum(counts[:, None], 1.0)
    variances = (
        shrink[:, None] * variances
        + (1.0 - shrink[:, None]) * global_var
        + 1e-4
    )

    predicted = np.argmax(ev.logits, axis=1)
    deviation = clr_ev - means[predicted]
    return np.mean((deviation * deviation) / variances[predicted], axis=1)

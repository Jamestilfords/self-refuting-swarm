import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "This measures whether the relative shape of an item's Dirichlet concentration vector matches the class-conditional evidence profile observed in training. Removing total concentration makes it distinct from vacuity and energy while retaining evidence-composition information."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.clip(ctx.h.alpha(tr.logits), 1e-12, 1e12)
    a_ev = np.clip(ctx.h.alpha(ev.logits), 1e-12, 1e12)

    # Centered log-ratio representation removes overall evidence magnitude
    # and retains only the Dirichlet concentration shape.
    z_tr = np.log(a_tr)
    z_tr = z_tr - z_tr.mean(axis=1, keepdims=True)
    z_ev = np.log(a_ev)
    z_ev = z_ev - z_ev.mean(axis=1, keepdims=True)

    y = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]
    dim = z_tr.shape[1]

    counts = np.bincount(y, minlength=n_classes).astype(float)
    global_mu = z_tr.mean(axis=0)
    global_var = z_tr.var(axis=0) + 1e-4

    sums = np.zeros((n_classes, dim), dtype=float)
    np.add.at(sums, y, z_tr)
    raw_mu = sums / np.maximum(counts[:, None], 1.0)

    # Shrink sparse class profiles toward the global evidence-shape profile.
    prior_strength = 10.0
    mu = (
        sums + prior_strength * global_mu[None, :]
    ) / (counts[:, None] + prior_strength)

    residual = z_tr - mu[y]
    sq_sums = np.zeros((n_classes, dim), dtype=float)
    np.add.at(sq_sums, y, residual * residual)
    raw_var = sq_sums / np.maximum(counts[:, None], 1.0)

    # Diagonal variance shrinkage avoids unstable estimates for small classes.
    var = (
        counts[:, None] * raw_var + prior_strength * global_var[None, :]
    ) / (counts[:, None] + prior_strength)
    var = np.maximum(var, 1e-4)

    pred = np.argmax(ev.logits, axis=1)
    delta = z_ev - mu[pred]
    return np.mean((delta * delta) / var[pred], axis=1)

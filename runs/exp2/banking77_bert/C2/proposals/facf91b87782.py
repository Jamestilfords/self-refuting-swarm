import numpy as np
from scipy.special import logsumexp

FAMILY = "density"
ARM = "ce"
RATIONALE = "A cosine-prototype mixture scores whether a normalized feature is supported by the full set of class directions, rather than only its nearest prototype. OOS inputs should have low aggregate von Mises-Fisher-style affinity to the in-scope class prototypes."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)

    labels = np.asarray(tr.labels)
    classes, inverse = np.unique(labels, return_inverse=True)
    n_classes = classes.size

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    np.add.at(prototypes, inverse, x_train)
    counts = np.bincount(inverse, minlength=n_classes).astype(float)
    prototypes = ctx.h.l2(prototypes)

    train_best = np.max(x_train @ prototypes.T, axis=1)
    concentration = 1.0 / max(1.0 - float(np.mean(train_best)), 0.02)
    concentration = float(np.clip(concentration, 2.0, 40.0))

    log_priors = np.log(counts / counts.sum())
    affinities = concentration * (x_eval @ prototypes.T) + log_priors
    return -logsumexp(affinities, axis=1)

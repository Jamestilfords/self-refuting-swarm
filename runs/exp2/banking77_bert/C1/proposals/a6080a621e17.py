import numpy as np
from scipy.special import logsumexp

FAMILY = "density"
ARM = "ce"
RATIONALE = "A cosine prototype free-energy score measures total angular support from all class directions rather than only the nearest prototype. OOS queries should have uniformly weak prototype affinity, producing higher negative free energy."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = tr.labels
    classes = np.unique(labels)

    prototypes = np.empty((classes.size, x_train.shape[1]), dtype=float)
    correct_cos = np.empty(x_train.shape[0], dtype=float)

    for j, cls in enumerate(classes):
        mask = labels == cls
        prototype = x_train[mask].mean(axis=0)
        prototype /= max(np.linalg.norm(prototype), 1e-12)
        prototypes[j] = prototype
        correct_cos[mask] = x_train[mask] @ prototype

    temperature = float(np.std(correct_cos))
    temperature = max(temperature, 0.03)

    similarities = x_eval @ prototypes.T
    return -temperature * logsumexp(similarities / temperature, axis=1)

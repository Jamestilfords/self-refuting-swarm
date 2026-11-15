import numpy as np
from scipy.special import logsumexp

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific angular prototype distributions model both alignment and within-class cosine concentration. OOS queries should have weak standardized similarity to every in-scope prototype rather than merely low raw similarity."
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
    means = np.empty(classes.size, dtype=float)
    scales = np.empty(classes.size, dtype=float)

    for i, c in enumerate(classes):
        members = x_train[labels == c]
        proto = members.mean(axis=0)
        proto /= max(np.linalg.norm(proto), 1e-12)
        prototypes[i] = proto

        train_cosine = members @ proto
        means[i] = train_cosine.mean()
        scales[i] = max(train_cosine.std(), 0.02)

    similarities = x_eval @ prototypes.T
    standardized = (similarities - means[None, :]) / scales[None, :]

    return -logsumexp(standardized, axis=1)

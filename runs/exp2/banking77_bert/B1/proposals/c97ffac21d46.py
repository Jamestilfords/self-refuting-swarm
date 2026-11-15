import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope queries should have high cosine affinity to the prototypes favored by their classifier distribution, whereas OOS inputs often have low prototype affinity even when their logit ranking is sharp. Softmax-weighting all prototype affinities avoids the maximum-cosine ranking duplicated by the earlier proposal."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0015"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)

    n_classes = tr.logits.shape[1]
    labels = tr.labels.astype(int)
    sums = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    np.add.at(sums, labels, x_train)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    prototypes = sums / np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    affinities = x_eval @ prototypes.T
    probs = ctx.h.softmax(ev.logits)
    expected_affinity = np.sum(probs * affinities, axis=1)

    return -expected_affinity

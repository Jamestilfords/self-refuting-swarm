import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope queries should align closely with at least one class prototype in normalized bottleneck feature space. OOS queries tend to have lower maximum cosine similarity to all learned class directions."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = np.asarray(tr.labels)
    classes, inverse = np.unique(labels, return_inverse=True)

    prototypes = np.zeros((classes.shape[0], x_train.shape[1]), dtype=float)
    np.add.at(prototypes, inverse, x_train)
    counts = np.bincount(inverse, minlength=classes.shape[0]).astype(float)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    max_cosine = np.full(x_eval.shape[0], -np.inf, dtype=float)
    block = 64
    for start in range(0, prototypes.shape[0], block):
        sims = x_eval @ prototypes[start:start + block].T
        max_cosine = np.maximum(max_cosine, np.max(sims, axis=1))

    return 1.0 - max_cosine

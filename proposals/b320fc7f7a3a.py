import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "L2-normalized class prototypes represent the angular directions of known intents independently of feature magnitude. OOS queries tend to have lower cosine similarity to every train-derived prototype."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0009"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)

    classes, inverse = np.unique(tr.labels, return_inverse=True)
    prototypes = np.zeros((classes.size, x_train.shape[1]), dtype=float)
    np.add.at(prototypes, inverse, x_train)
    counts = np.bincount(inverse, minlength=classes.size).astype(float)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    max_cosine = np.max(x_eval @ prototypes.T, axis=1)
    return 1.0 - max_cosine

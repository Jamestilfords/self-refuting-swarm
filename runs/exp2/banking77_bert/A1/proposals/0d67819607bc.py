import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "L2-normalized class prototypes measure angular support for each query, making OOS items with weak semantic alignment to every intent score highly. Averaging each class prototype from training data makes the score robust to individual examples."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = tr.labels
    n_classes = int(np.max(labels)) + 1

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)

    norms = np.sqrt(np.sum(prototypes * prototypes, axis=1, keepdims=True))
    prototypes /= np.maximum(norms, 1e-12)

    best_similarity = np.max(x_eval @ prototypes.T, axis=1)
    return 1.0 - best_similarity

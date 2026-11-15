import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "In-scope queries should align strongly with at least one class prototype in normalized bottleneck feature space. Low maximum cosine similarity indicates that an item lies away from all learned intent directions."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)

    labels = tr.labels.astype(int)
    n_classes = int(np.max(labels)) + 1
    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)

    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    max_cosine = np.max(x_eval @ prototypes.T, axis=1)
    return 1.0 - max_cosine

import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "L2-normalized pooled representations of in-scope queries should align with their class prototype. "
    "Low maximum cosine similarity indicates that a query has weak directional support from every known intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0075"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_x = ctx.h.l2(tr.pooled)
    eval_x = ctx.h.l2(ev.pooled)
    labels = tr.labels
    n_classes = int(np.max(labels)) + 1

    prototypes = np.zeros((n_classes, train_x.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, train_x)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    max_cosine = np.max(eval_x @ prototypes.T, axis=1)
    return 1.0 - max_cosine

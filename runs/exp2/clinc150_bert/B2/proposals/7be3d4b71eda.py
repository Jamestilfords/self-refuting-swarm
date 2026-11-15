import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope utterances should align strongly with at least one train-derived intent prototype "
    "in normalized pooled representation space, while OOS utterances have lower maximum cosine similarity."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0031"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    labels = tr.labels
    classes = np.unique(labels)

    prototypes = np.empty((classes.shape[0], x_train.shape[1]), dtype=float)
    for i, cls in enumerate(classes):
        proto = np.mean(x_train[labels == cls], axis=0)
        norm = np.linalg.norm(proto)
        if norm > 0:
            prototypes[i] = proto / norm
        else:
            prototypes[i] = proto

    max_cosine = np.max(x_eval @ prototypes.T, axis=1)
    return 1.0 - max_cosine

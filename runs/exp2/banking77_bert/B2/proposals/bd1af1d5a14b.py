import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope representations should align closely with at least one normalized intent prototype, "
    "whereas OOS queries are expected to have weaker maximum cosine support from all known intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0059"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    labels = np.asarray(tr.labels)
    n_classes = int(np.max(labels)) + 1

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    max_cosine = np.max(x_eval @ prototypes.T, axis=1)
    return 1.0 - max_cosine

import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope examples should have high cosine similarity to at least one class prototype in normalized bottleneck-feature space. OOS queries are expected to be poorly aligned with every learned intent direction."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0005"]


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

    max_similarity = np.max(x_eval @ prototypes.T, axis=1)
    return -max_similarity

import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "L2-normalized class prototypes measure angular compatibility with in-scope intents, "
    "so low maximum cosine similarity identifies queries whose feature direction is atypical."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = tr.labels
    n_classes = tr.logits.shape[1]

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    max_similarity = np.max(x_eval @ prototypes.T, axis=1)
    return 1.0 - max_similarity

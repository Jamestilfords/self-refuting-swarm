import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "L2-normalized class prototypes measure angular support for the classifier-predicted intent, "
    "so OOS queries with semantically misaligned representations receive high scores."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0058", "m_0062"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = tr.labels.astype(int)
    n_classes = tr.logits.shape[1]

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    predicted = np.argmax(ev.logits, axis=1)
    similarity = np.sum(x_eval * prototypes[predicted], axis=1)
    return 1.0 - np.clip(similarity, -1.0, 1.0)

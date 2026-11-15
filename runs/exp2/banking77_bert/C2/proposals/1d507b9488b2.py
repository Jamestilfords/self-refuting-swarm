import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "In-scope queries should be directionally consistent with the prototype of the class predicted by the classifier. OOS queries may have a high cosine match to some class by chance but weaker agreement with their predicted class prototype."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


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
    cosine = np.sum(x_eval * prototypes[predicted], axis=1)
    return 1.0 - cosine

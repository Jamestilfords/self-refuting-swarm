import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope queries should have high cosine similarity to the prototype of their predicted intent "
    "in normalized encoder space; low predicted-prototype similarity indicates unsupported intent geometry."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0141"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_x = ctx.h.l2(tr.pooled)
    eval_x = ctx.h.l2(ev.pooled)
    labels = tr.labels.astype(int)
    n_classes = tr.logits.shape[1]

    prototypes = np.zeros((n_classes, train_x.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, train_x)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    pred = np.argmax(ev.logits, axis=1)
    similarity = np.sum(eval_x * prototypes[pred], axis=1)
    return 1.0 - similarity

import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Known intents form characteristic directions in l2-normalized bottleneck space. "
    "Low cosine similarity to the prototype of the classifier-predicted intent indicates representation-logit disagreement and likely OOS input."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0069", "m_0070"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = tr.labels
    k = tr.logits.shape[1]

    prototypes = np.zeros((k, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=k).astype(float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    predicted = np.argmax(ev.logits, axis=1)
    similarity = np.sum(x_eval * prototypes[predicted], axis=1)
    return 1.0 - similarity

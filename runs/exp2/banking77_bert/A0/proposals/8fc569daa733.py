import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "The classifier-selected intent should have a prototype direction aligned with an in-scope query. "
    "Low cosine similarity to the predicted class prototype identifies representation-classifier disagreement indicative of OOS inputs."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)

    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]
    prototypes = np.zeros((n_classes, dim), dtype=float)
    counts = np.bincount(tr.labels, minlength=n_classes).astype(float)

    np.add.at(prototypes, tr.labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)

    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes /= np.maximum(norms, 1e-12)

    predicted = np.argmax(ev.logits, axis=1)
    assigned_prototypes = prototypes[predicted]
    similarity = np.sum(x_eval * assigned_prototypes, axis=1)

    missing = counts[predicted] == 0
    if np.any(missing):
        similarity[missing] = 0.0

    return 1.0 - similarity

import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope representations should align closely with the normalized prototype of their classifier-predicted intent. OOS queries can be assigned confidently while having weak cosine agreement with that intent's typical feature direction."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0079", "m_0095"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)

    labels = tr.labels.astype(int)
    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]

    sums = np.zeros((n_classes, dim), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(sums, labels, x_train)

    prototypes = sums / np.maximum(counts[:, None], 1.0)
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes = prototypes / np.maximum(norms, 1e-12)

    predicted = np.argmax(ev.logits, axis=1)
    cosine = np.sum(x_eval * prototypes[predicted], axis=1)
    return 1.0 - np.clip(cosine, -1.0, 1.0)

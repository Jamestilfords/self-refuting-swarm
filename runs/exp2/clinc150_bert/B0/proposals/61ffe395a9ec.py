import numpy as np

FAMILY = "logit"
ARM = "edl"
RATIONALE = (
    "The normalized Dirichlet concentration vector describes how evidence is allocated among intents, "
    "independently of total concentration. OOS queries can have evidence-allocation patterns atypical of "
    "the in-scope class they most resemble, even when vacuity or energy alone is not decisive."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-12)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-12)

    p_tr = a_tr / np.sum(a_tr, axis=1, keepdims=True)
    p_ev = a_ev / np.sum(a_ev, axis=1, keepdims=True)

    n_classes = p_tr.shape[1]
    labels = np.asarray(tr.labels, dtype=int)
    counts = np.bincount(labels, minlength=n_classes).astype(float)

    prototypes = np.zeros((n_classes, n_classes), dtype=float)
    np.add.at(prototypes, labels, p_tr)
    global_proto = np.mean(p_tr, axis=0)
    valid = counts > 0
    prototypes[valid] /= counts[valid, None]
    prototypes[~valid] = global_proto

    train_proto = prototypes[labels]
    train_dev = np.sum(np.abs(p_tr - train_proto), axis=1)

    scales = np.zeros(n_classes, dtype=float)
    np.add.at(scales, labels, train_dev)
    scales[valid] /= counts[valid]
    fallback = max(float(np.median(train_dev)), 1e-6)
    scales[~valid] = fallback
    scales = np.maximum(scales, max(fallback * 0.05, 1e-6))

    predicted = np.argmax(a_ev, axis=1)
    deviation = np.sum(np.abs(p_ev - prototypes[predicted]), axis=1)
    return np.nan_to_num(deviation / scales[predicted], nan=0.0, posinf=1e12, neginf=0.0)

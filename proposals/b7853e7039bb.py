import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query should be angularly aligned with the prototype of the intent selected by its classifier logits. "
    "Class-specific train-fitted cosine calibration detects weak predicted-intent support without reducing to nearest-prototype similarity."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0020", "m_0075", "m_0099"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = ctx.h.l2(tr.feats)
    xev = ctx.h.l2(ev.feats)
    k = tr.logits.shape[1]

    prototypes = np.zeros((k, xtr.shape[1]), dtype=float)
    counts = np.bincount(tr.labels, minlength=k).astype(float)
    np.add.at(prototypes, tr.labels, xtr)

    global_proto = np.mean(xtr, axis=0, keepdims=True)
    missing = counts == 0
    prototypes[~missing] /= counts[~missing, None]
    prototypes[missing] = global_proto
    prototypes = ctx.h.l2(prototypes)

    train_pred = np.argmax(tr.logits, axis=1)
    train_cos = np.sum(xtr * prototypes[train_pred], axis=1)

    sums = np.bincount(train_pred, weights=train_cos, minlength=k)
    pred_counts = np.bincount(train_pred, minlength=k).astype(float)
    means = sums / np.maximum(pred_counts, 1.0)

    centered = train_cos - means[train_pred]
    vars_ = np.bincount(
        train_pred, weights=centered * centered, minlength=k
    ) / np.maximum(pred_counts, 1.0)
    stds = np.sqrt(np.maximum(vars_, 1e-6))

    fallback_std = np.std(train_cos) + 1e-6
    stds[pred_counts < 2] = fallback_std

    eval_pred = np.argmax(ev.logits, axis=1)
    eval_cos = np.sum(xev * prototypes[eval_pred], axis=1)

    return (means[eval_pred] - eval_cos) / stds[eval_pred]

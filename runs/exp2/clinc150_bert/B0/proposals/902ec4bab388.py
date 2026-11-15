import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Known-intent means define classifier-discriminative directions, so OOS queries should devote "
    "an unusually large fraction of their feature energy to the orthogonal remainder. Calibrating "
    "this remainder fraction within the logit-predicted intent avoids treating naturally diffuse intents as OOS."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    n_classes = int(max(np.max(y), tr.logits.shape[1] - 1)) + 1
    counts = np.bincount(y, minlength=n_classes).astype(float)
    means = np.zeros((n_classes, x.shape[1]), dtype=float)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    center = np.mean(x, axis=0)
    between = (means - center) * np.sqrt(np.maximum(counts[:, None], 1.0))
    _, singular_values, vt = np.linalg.svd(between, full_matrices=False)

    if singular_values.size:
        tol = max(between.shape) * np.finfo(float).eps * singular_values[0]
        rank = int(np.sum(singular_values > tol))
    else:
        rank = 0

    def remainder_fraction(a):
        centered = a - center
        total = np.sum(centered * centered, axis=1)
        if rank > 0:
            basis = vt[:rank]
            projected = centered @ basis.T
            explained = np.sum(projected * projected, axis=1)
        else:
            explained = np.zeros(a.shape[0], dtype=float)
        remainder = np.maximum(total - explained, 0.0)
        return remainder / np.maximum(total, 1e-12)

    train_fraction = remainder_fraction(x)
    eval_fraction = remainder_fraction(z)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    out = np.empty(z.shape[0], dtype=float)

    global_sorted = np.sort(train_fraction)
    for c in np.unique(eval_pred):
        mask_eval = eval_pred == c
        mask_train = train_pred == c
        reference = np.sort(train_fraction[mask_train])
        if reference.size < 8:
            reference = global_sorted
        ranks = np.searchsorted(reference, eval_fraction[mask_eval], side="left")
        tail = (reference.size - ranks + 0.5) / (reference.size + 1.0)
        out[mask_eval] = -np.log(np.maximum(tail, 1e-12))

    return out

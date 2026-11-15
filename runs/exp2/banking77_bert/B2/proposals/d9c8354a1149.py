import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "The span of class means captures directions used to discriminate known intents. "
    "After removing that supervised subspace, unusually large residual feature energy indicates content unsupported by the intent taxonomy."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0029"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)

    mean = x.mean(axis=0)
    n_classes = tr.logits.shape[1]
    class_means = np.empty((n_classes, x.shape[1]), dtype=np.float64)
    for c in range(n_classes):
        members = x[y == c]
        class_means[c] = members.mean(axis=0) if len(members) else mean

    between = class_means - class_means.mean(axis=0)
    _, singular, vt = np.linalg.svd(between, full_matrices=False)
    tol = max(between.shape) * np.finfo(np.float64).eps * (singular[0] if len(singular) else 1.0)
    rank = int(np.sum(singular > tol))

    def residual_fraction(a):
        centered = a - mean
        total = np.sum(centered * centered, axis=1)
        if rank:
            coords = centered @ vt[:rank].T
            explained = np.sum(coords * coords, axis=1)
            residual = np.maximum(total - explained, 0.0)
        else:
            residual = total
        return np.log1p(residual) - np.log1p(total + 1e-12)

    train_raw = residual_fraction(x)
    eval_raw = residual_fraction(z)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_med = np.median(train_raw)
    global_mad = np.median(np.abs(train_raw - global_med)) + 1e-6
    scores = np.empty(len(z), dtype=np.float64)

    for c in range(n_classes):
        mask = train_pred == c
        if np.sum(mask) >= 8:
            values = train_raw[mask]
            med = np.median(values)
            mad = np.median(np.abs(values - med)) + 1e-6
        else:
            med, mad = global_med, global_mad
        out = eval_pred == c
        scores[out] = (eval_raw[out] - med) / mad

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

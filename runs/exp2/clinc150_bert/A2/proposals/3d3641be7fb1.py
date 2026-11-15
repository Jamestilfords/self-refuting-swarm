import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Known intents should concentrate a substantial fraction of their bottleneck representation "
    "in directions that separate class means. A high class-calibrated fraction of energy remaining "
    "outside this discriminative subspace indicates variation unsupported by the learned intent structure."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0087"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]

    global_mean = np.mean(x, axis=0)
    means = np.empty((n_classes, x.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)

    for c in range(n_classes):
        mask = labels == c
        if np.any(mask):
            means[c] = np.mean(x[mask], axis=0)
        else:
            means[c] = global_mean

    # Weight class-mean directions by available evidence, then retain only the
    # strongest supervised directions rather than an arbitrary full-rank basis.
    between = (means - global_mean) * np.sqrt(np.maximum(counts, 1.0))[:, None]
    _, singular, vh = np.linalg.svd(between, full_matrices=False)
    if singular.size and singular[0] > 0:
        keep = singular > singular[0] * 1e-8
        rank = min(64, int(np.sum(keep)))
    else:
        rank = 0
    basis = vh[:rank].T

    def residual_fraction(a):
        centered = a - global_mean
        total = np.sum(centered * centered, axis=1)
        if rank:
            projected = centered @ basis
            explained = np.sum(projected * projected, axis=1)
        else:
            explained = np.zeros(a.shape[0], dtype=float)
        return np.maximum(total - explained, 0.0) / np.maximum(total, 1e-12)

    train_fraction = residual_fraction(x)
    eval_fraction = residual_fraction(z)

    # Predicted-intent calibration prevents broad or weakly separated intents
    # from being systematically assigned inflated residual fractions.
    predicted_train = np.argmax(tr.logits, axis=1)
    predicted_eval = np.argmax(ev.logits, axis=1)
    global_scale = max(float(np.quantile(train_fraction, 0.90)), 1e-8)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_fraction[predicted_train == c]
        if values.size >= 5:
            scales[c] = max(float(np.quantile(values, 0.90)), 1e-8)

    scores = np.log1p(eval_fraction / scales[predicted_eval])
    return np.nan_to_num(scores, nan=0.0, posinf=50.0, neginf=0.0)

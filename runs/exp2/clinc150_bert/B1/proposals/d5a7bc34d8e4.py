import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Trailing within-class eigen-directions represent low-variance constraints of each known intent. "
    "Class-conditioned normalization makes violations of those constraints comparable across intents without collapsing to a global residual ranking."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    projector = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    train_tail = np.asarray(projector.bottom(tr.pooled), dtype=float).reshape(-1)
    eval_tail = np.asarray(projector.bottom(ev.pooled), dtype=float).reshape(-1)

    train_tail = np.nan_to_num(train_tail, nan=0.0, posinf=1e12, neginf=0.0)
    eval_tail = np.nan_to_num(eval_tail, nan=0.0, posinf=1e12, neginf=0.0)
    train_tail = np.maximum(train_tail, 0.0)
    eval_tail = np.maximum(eval_tail, 0.0)

    train_value = np.log1p(train_tail)
    eval_value = np.log1p(eval_tail)

    labels = np.asarray(tr.labels, dtype=int)
    pred = np.argmax(np.asarray(ev.logits), axis=1).astype(int)
    n_classes = int(max(np.max(labels), np.max(pred)) + 1)

    global_med = float(np.median(train_value))
    global_mad = float(np.median(np.abs(train_value - global_med)))
    global_scale = max(1.4826 * global_mad, 1e-6)

    medians = np.full(n_classes, global_med, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_value[labels == c]
        if values.size >= 4:
            med = float(np.median(values))
            mad = float(np.median(np.abs(values - med)))
            medians[c] = med
            scales[c] = max(1.4826 * mad, 0.15 * global_scale, 1e-6)

    clipped_pred = np.clip(pred, 0, n_classes - 1)
    z_tail = (eval_value - medians[clipped_pred]) / scales[clipped_pred]

    # A smooth upper-tail transform retains magnitude differences beyond the
    # in-scope range rather than reducing the detector to a percentile ranking.
    return np.logaddexp(0.0, z_tail)

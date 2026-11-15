import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "In-scope representations should usually have a clearly preferred Gaussian intent, "
    "whereas OOS queries often lie near a boundary between multiple intent regions. "
    "A predicted-intent-conditioned distance-gap calibration accounts for naturally ambiguous intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_order = np.partition(train_d, 1, axis=1)
    eval_order = np.partition(eval_d, 1, axis=1)
    train_gap = train_order[:, 1] - train_order[:, 0]
    eval_gap = eval_order[:, 1] - eval_order[:, 0]

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_center = np.median(train_gap)
    global_scale = np.median(np.abs(train_gap - global_center)) * 1.4826
    global_scale = max(float(global_scale), 1e-6)

    centers = np.full(tr.logits.shape[1], global_center, dtype=float)
    scales = np.full(tr.logits.shape[1], global_scale, dtype=float)

    for c in range(tr.logits.shape[1]):
        values = train_gap[train_pred == c]
        if values.size >= 8:
            center = np.median(values)
            scale = np.median(np.abs(values - center)) * 1.4826
            centers[c] = center
            scales[c] = max(float(scale), 1e-6)

    return (centers[eval_pred] - eval_gap) / scales[eval_pred]

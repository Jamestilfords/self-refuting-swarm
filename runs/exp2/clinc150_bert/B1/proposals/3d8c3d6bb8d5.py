import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "OOS queries can have a small top-two separation relative to the overall logit dispersion, "
    "even when their absolute energy is high. Class-conditional train calibration accounts for "
    "different natural margin scales among routed intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def normalized_margin(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        dispersion = np.sqrt(np.mean(centered * centered, axis=1) + 1e-12)
        return margin / dispersion

    train_margin = normalized_margin(tr.logits)
    eval_margin = normalized_margin(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_center = np.median(train_margin)
    global_scale = np.median(np.abs(train_margin - global_center))
    global_scale = max(1.4826 * global_scale, 1e-6)

    n_classes = tr.logits.shape[1]
    centers = np.full(n_classes, global_center, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_margin[train_pred == c]
        if values.size >= 4:
            center = np.median(values)
            scale = 1.4826 * np.median(np.abs(values - center))
            shrink = values.size / (values.size + 10.0)
            centers[c] = shrink * center + (1.0 - shrink) * global_center
            scales[c] = max(
                shrink * max(scale, 1e-6) + (1.0 - shrink) * global_scale,
                1e-6,
            )

    return (centers[eval_pred] - eval_margin) / scales[eval_pred]

import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "OOS queries often have a weaker separation between their two most plausible intents after removing logit scale and offset. Class-conditional calibration makes the normalized top-two margin comparable across intents with different inherent confidence levels."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def normalized_margin(logits):
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        top2 = np.partition(centered, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        scale = np.sqrt(np.mean(centered * centered, axis=1))
        return margin / np.maximum(scale, 1e-8)

    train_margin = normalized_margin(tr.logits)
    eval_margin = normalized_margin(ev.logits)
    n_classes = tr.logits.shape[1]

    global_med = np.median(train_margin)
    global_mad = np.median(np.abs(train_margin - global_med))
    global_scale = max(1.4826 * global_mad, 1e-6)

    centers = np.full(n_classes, global_med, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_margin[tr.labels == c]
        if values.size >= 4:
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            centers[c] = med
            scales[c] = max(1.4826 * mad, global_scale * 0.1, 1e-6)

    predicted = np.argmax(ev.logits, axis=1)
    z = (eval_margin - centers[predicted]) / scales[predicted]
    return np.clip(-z, -50.0, 50.0)

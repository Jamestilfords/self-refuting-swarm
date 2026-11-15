import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "In-scope examples typically exhibit a class-specific dominant top logit relative to the "
    "overall logit spread. OOS queries often have a weaker normalized top-two separation even "
    "when their raw maximum probability or energy appears confident."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def normalized_margin(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        spread = np.sqrt(np.mean(centered * centered, axis=1))
        return margin / (spread + 1e-6)

    train_shape = normalized_margin(tr.logits)
    eval_shape = normalized_margin(ev.logits)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_med = np.median(train_shape)
    global_mad = np.median(np.abs(train_shape - global_med))
    global_scale = max(1.4826 * global_mad, np.std(train_shape), 1e-6)

    medians = np.full(n_classes, global_med, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_shape[train_pred == c]
        if values.size >= 4:
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            medians[c] = med
            scales[c] = max(1.4826 * mad, global_scale * 0.05, 1e-6)

    return (medians[eval_pred] - eval_shape) / scales[eval_pred]

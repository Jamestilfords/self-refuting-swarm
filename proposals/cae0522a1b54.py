import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "OOS queries often have a smaller top-two separation after accounting for their overall "
    "logit scale. Class-conditional train calibration avoids treating naturally ambiguous "
    "known intents as OOS merely because they have lower margins."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0008"]


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
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_med = np.median(train_margin)
    global_q25, global_q75 = np.percentile(train_margin, [25.0, 75.0])
    global_scale = max(global_q75 - global_q25, 1e-4)

    medians = np.full(n_classes, global_med, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for cls in range(n_classes):
        values = train_margin[train_pred == cls]
        if values.size >= 8:
            medians[cls] = np.median(values)
            q25, q75 = np.percentile(values, [25.0, 75.0])
            scales[cls] = max(q75 - q25, 1e-4)

    z = (medians[eval_pred] - eval_margin) / scales[eval_pred]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

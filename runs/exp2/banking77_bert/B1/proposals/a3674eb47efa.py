import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "The normalized top-two logit margin measures the shape of classifier evidence beyond "
    "energy or maximum probability. Calibrating this margin within the predicted intent "
    "accounts for intents that naturally have different decision-boundary sharpness."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0057"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def normalized_margin(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        norm = np.sqrt(np.sum(logits * logits, axis=1))
        return margin / np.maximum(norm, 1e-8)

    train_margin = normalized_margin(tr.logits)
    eval_margin = normalized_margin(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_med = np.median(train_margin)
    global_q25, global_q75 = np.quantile(train_margin, [0.25, 0.75])
    global_scale = max(global_q75 - global_q25, 1e-6)

    n_classes = tr.logits.shape[1]
    medians = np.full(n_classes, global_med, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_margin[train_pred == c]
        if values.size >= 8:
            q25, med, q75 = np.quantile(values, [0.25, 0.5, 0.75])
            medians[c] = med
            scales[c] = max(q75 - q25, 1e-6)

    scores = (medians[eval_pred] - eval_margin) / scales[eval_pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

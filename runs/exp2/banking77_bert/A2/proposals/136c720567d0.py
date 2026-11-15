import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "OOS queries often have a weak separation between the two most competitive known intents. "
    "Predicted-class train calibration makes top-two margins comparable across intents with different intrinsic confidence scales."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0101"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_order = np.partition(tr.logits, -2, axis=1)
    eval_order = np.partition(ev.logits, -2, axis=1)
    train_margin = train_order[:, -1] - train_order[:, -2]
    eval_margin = eval_order[:, -1] - eval_order[:, -2]

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]

    centers = np.zeros(k, dtype=float)
    scales = np.ones(k, dtype=float)
    global_center = np.median(train_margin)
    global_scale = np.median(np.abs(train_margin - global_center)) + 1e-6

    for c in range(k):
        values = train_margin[train_pred == c]
        if values.size >= 4:
            center = np.median(values)
            scale = np.median(np.abs(values - center))
            centers[c] = center
            scales[c] = max(scale, 0.1 * global_scale, 1e-6)
        else:
            centers[c] = global_center
            scales[c] = global_scale

    score_values = (centers[eval_pred] - eval_margin) / scales[eval_pred]
    return np.nan_to_num(score_values, nan=0.0, posinf=1e6, neginf=-1e6)

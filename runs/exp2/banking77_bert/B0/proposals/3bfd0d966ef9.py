import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Top-two logit margins measure ambiguity between the predicted intent and its closest competitor. Calibrating margins within the classifier-predicted intent accounts for classes with intrinsically different decision-boundary sharpness."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def margin_and_pred(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        pred = np.argmax(logits, axis=1)
        return margin, pred

    tr_margin, tr_pred = margin_and_pred(tr.logits)
    ev_margin, ev_pred = margin_and_pred(ev.logits)
    k = tr.logits.shape[1]

    centers = np.empty(k, dtype=float)
    scales = np.empty(k, dtype=float)
    global_center = np.median(tr_margin)
    global_scale = np.median(np.abs(tr_margin - global_center))
    global_scale = max(float(global_scale), 1e-6)

    for c in range(k):
        values = tr_margin[tr_pred == c]
        if values.size < 4:
            centers[c] = global_center
            scales[c] = global_scale
        else:
            center = np.median(values)
            scale = np.median(np.abs(values - center))
            centers[c] = center
            scales[c] = max(float(scale), 1e-6)

    return (centers[ev_pred] - ev_margin) / scales[ev_pred]

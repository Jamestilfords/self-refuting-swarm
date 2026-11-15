import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "OOS inputs often produce less decisive separation between the largest and second-largest logits. Calibrating the top-two margin within each predicted class accounts for intent-specific confidence scales using training logits only."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def margin(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    train_margin = margin(tr.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_margin = margin(ev.logits)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_center = np.median(train_margin)
    global_scale = np.median(np.abs(train_margin - global_center)) * 1.4826
    global_scale = max(float(global_scale), 1e-6)

    centers = np.full(tr.logits.shape[1], global_center, dtype=float)
    scales = np.full(tr.logits.shape[1], global_scale, dtype=float)

    for cls in range(tr.logits.shape[1]):
        values = train_margin[train_pred == cls]
        if values.size >= 8:
            center = np.median(values)
            scale = np.median(np.abs(values - center)) * 1.4826
            centers[cls] = center
            scales[cls] = max(float(scale), global_scale * 0.1, 1e-6)

    return (centers[eval_pred] - eval_margin) / scales[eval_pred]

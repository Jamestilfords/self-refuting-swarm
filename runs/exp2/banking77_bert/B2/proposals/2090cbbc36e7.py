import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope predictions tend to exhibit a class-specific separation between the most likely and runner-up intents, whereas OOS inputs often have a smaller decision margin. Calibrating the negative top-two margin by the predicted class accounts for different intrinsic class confusability."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def top_two_margin(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    train_margin = top_two_margin(tr.logits)
    eval_margin = top_two_margin(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_center = np.median(train_margin)
    global_scale = np.median(np.abs(train_margin - global_center)) * 1.4826
    global_scale = max(float(global_scale), 1e-6)

    n_classes = tr.logits.shape[1]
    centers = np.full(n_classes, global_center, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_margin[train_pred == c]
        if values.size >= 4:
            center = np.median(values)
            scale = np.median(np.abs(values - center)) * 1.4826
            centers[c] = center
            scales[c] = max(float(scale), 1e-6)

    scores = (centers[eval_pred] - eval_margin) / scales[eval_pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "OOS inputs often have a smaller separation between the two most competitive intent logits than in-scope examples. Class-conditional robust calibration accounts for intents with naturally ambiguous decision boundaries."
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

    global_med = np.median(train_margin)
    global_scale = np.median(np.abs(train_margin - global_med)) * 1.4826
    global_scale = max(float(global_scale), 1e-6)

    n_classes = tr.logits.shape[1]
    medians = np.full(n_classes, global_med, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_margin[train_pred == c]
        if values.size >= 8:
            med = np.median(values)
            scale = np.median(np.abs(values - med)) * 1.4826
            medians[c] = med
            scales[c] = max(float(scale), 1e-6)

    scores = (medians[eval_pred] - eval_margin) / scales[eval_pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Class-conditioned centered-logit norm measures whether an input has the typical discriminative logit scale of the intent it is predicted to belong to. OOS queries often produce abnormally weak class separation even when their top predicted class is plausible."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_logits = np.asarray(tr.logits, dtype=float)
    eval_logits = np.asarray(ev.logits, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    train_centered = train_logits - train_logits.mean(axis=1, keepdims=True)
    eval_centered = eval_logits - eval_logits.mean(axis=1, keepdims=True)
    train_norm = np.sqrt(np.sum(train_centered * train_centered, axis=1))
    eval_norm = np.sqrt(np.sum(eval_centered * eval_centered, axis=1))

    n_classes = train_logits.shape[1]
    medians = np.empty(n_classes, dtype=float)
    scales = np.empty(n_classes, dtype=float)
    global_median = np.median(train_norm)
    global_scale = np.median(np.abs(train_norm - global_median)) + 1e-6

    for c in range(n_classes):
        values = train_norm[labels == c]
        if values.size:
            medians[c] = np.median(values)
            scales[c] = np.median(np.abs(values - medians[c])) + 1e-6
        else:
            medians[c] = global_median
            scales[c] = global_scale

    predicted = np.argmax(eval_logits, axis=1)
    deficit = (medians[predicted] - eval_norm) / scales[predicted]
    return np.clip(deficit, -100.0, 100.0)

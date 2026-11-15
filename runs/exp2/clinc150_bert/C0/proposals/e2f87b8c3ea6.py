import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "The centered logit norm measures prediction sharpness independently of a common logit offset. OOS inputs can have unusually weak or unusually extreme sharpness relative to the in-scope examples of their predicted class."
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
    global_med = np.median(train_norm)
    global_mad = np.median(np.abs(train_norm - global_med))
    global_scale = max(1.4826 * global_mad, 1e-6)

    medians = np.full(n_classes, global_med, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)
    for c in range(n_classes):
        values = train_norm[labels == c]
        if values.size >= 3:
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            medians[c] = med
            scales[c] = max(1.4826 * mad, 1e-6)

    predicted = np.argmax(eval_logits, axis=1)
    scores = np.abs(eval_norm - medians[predicted]) / scales[predicted]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)

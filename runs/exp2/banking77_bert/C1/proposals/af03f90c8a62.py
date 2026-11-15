import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Known-intent examples tend to have class-dependent logit norms established during training, while OOS queries often produce comparatively weak evidence across all classes. A predicted-class-conditioned robust normalization avoids conflating naturally different confidence scales between intent classes."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_logits = np.asarray(tr.logits, dtype=float)
    eval_logits = np.asarray(ev.logits, dtype=float)

    train_pred = np.argmax(train_logits, axis=1)
    eval_pred = np.argmax(eval_logits, axis=1)
    train_norm = np.sqrt(np.sum(train_logits * train_logits, axis=1))
    eval_norm = np.sqrt(np.sum(eval_logits * eval_logits, axis=1))

    global_med = np.median(train_norm)
    global_mad = np.median(np.abs(train_norm - global_med))
    global_scale = max(1.4826 * global_mad, 1e-6)

    n_classes = train_logits.shape[1]
    medians = np.full(n_classes, global_med, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_norm[train_pred == c]
        if values.size >= 4:
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            medians[c] = med
            scales[c] = max(1.4826 * mad, 0.1 * global_scale, 1e-6)

    return (medians[eval_pred] - eval_norm) / scales[eval_pred]

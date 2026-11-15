import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Class-mean directions contain the representation components most useful for separating known intents. "
    "After removing that discriminative subspace, OOS queries should retain unusually large residual energy relative to train-fitted, predicted-intent-specific residual structure."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    global_mean = x_train.mean(axis=0)
    n_classes = int(max(labels.max(), tr.logits.shape[1])) + 1

    class_means = np.empty((n_classes, x_train.shape[1]), dtype=float)
    for c in range(n_classes):
        members = x_train[labels == c]
        class_means[c] = members.mean(axis=0) if len(members) else global_mean

    mean_offsets = class_means - global_mean
    _, singular_values, vt = np.linalg.svd(mean_offsets, full_matrices=False)
    if singular_values.size:
        keep = singular_values > (singular_values[0] * 1e-8)
        basis = vt[keep]
    else:
        basis = np.empty((0, x_train.shape[1]), dtype=float)

    def residual_ratio(x):
        centered = x - global_mean
        total = np.sum(centered * centered, axis=1)
        if basis.shape[0]:
            projected = centered @ basis.T
            explained = np.sum(projected * projected, axis=1)
        else:
            explained = np.zeros_like(total)
        residual = np.maximum(total - explained, 0.0)
        return residual / (total + 1e-8)

    train_ratio = residual_ratio(x_train)
    eval_ratio = residual_ratio(x_eval)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_med = np.median(train_ratio)
    global_scale = np.median(np.abs(train_ratio - global_med)) * 1.4826 + 1e-6
    med = np.full(tr.logits.shape[1], global_med, dtype=float)
    scale = np.full(tr.logits.shape[1], global_scale, dtype=float)

    for c in range(tr.logits.shape[1]):
        values = train_ratio[train_pred == c]
        if values.size >= 8:
            m = np.median(values)
            s = np.median(np.abs(values - m)) * 1.4826
            med[c] = m
            scale[c] = max(s, global_scale * 0.1, 1e-6)

    return (eval_ratio - med[eval_pred]) / scale[eval_pred]

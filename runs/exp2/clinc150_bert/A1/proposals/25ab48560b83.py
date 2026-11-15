import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Per-intent diagonal variances capture dimensions that are tightly constrained for one intent "
    "but variable for another. Predicted-class shrinkage Mahalanobis surprise flags samples that "
    "are incompatible with the classifier-routed intent without unstable full covariance inversion."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0107", "m_0119"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels)
    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)

    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]
    global_mean = x_train.mean(axis=0)
    global_var = x_train.var(axis=0) + 1e-6

    means = np.empty((n_classes, dim), dtype=np.float64)
    variances = np.empty((n_classes, dim), dtype=np.float64)

    for c in range(n_classes):
        xc = x_train[y == c]
        if len(xc) < 2:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = xc.mean(axis=0)
            class_var = xc.var(axis=0)
            shrink = dim / (dim + len(xc))
            variances[c] = (1.0 - shrink) * class_var + shrink * global_var + 1e-6

    train_diff = x_train - means[pred_train]
    train_score = np.mean(
        (train_diff * train_diff) / variances[pred_train],
        axis=1,
    )

    class_center = np.empty(n_classes, dtype=np.float64)
    class_scale = np.empty(n_classes, dtype=np.float64)
    overall_center = np.median(train_score)
    overall_scale = np.median(np.abs(train_score - overall_center)) + 1e-6

    for c in range(n_classes):
        values = train_score[pred_train == c]
        if len(values) < 5:
            class_center[c] = overall_center
            class_scale[c] = overall_scale
        else:
            center = np.median(values)
            class_center[c] = center
            class_scale[c] = np.median(np.abs(values - center)) + 1e-6

    eval_diff = x_eval - means[pred_eval]
    eval_score = np.mean(
        (eval_diff * eval_diff) / variances[pred_eval],
        axis=1,
    )
    return (eval_score - class_center[pred_eval]) / class_scale[pred_eval]

import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Per-intent diagonal variances retain class-specific semantic tolerances that are lost "
    "under a tied covariance. OOS queries should have unusually large standardized residuals "
    "relative to the class selected by the classifier."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0035", "m_0037"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)
    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)

    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]
    global_mean = x_train.mean(axis=0)
    global_var = x_train.var(axis=0) + 1e-6

    means = np.empty((n_classes, dim), dtype=np.float64)
    variances = np.empty((n_classes, dim), dtype=np.float64)
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)

    for c in range(n_classes):
        xc = x_train[labels == c]
        if xc.shape[0] == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = xc.mean(axis=0)
            class_var = xc.var(axis=0)
            shrink = dim / (dim + xc.shape[0])
            variances[c] = (1.0 - shrink) * class_var + shrink * global_var + 1e-6

    train_residual = (
        ((x_train - means[pred_train]) ** 2) / variances[pred_train]
    ).mean(axis=1)

    class_center = np.zeros(n_classes, dtype=np.float64)
    class_scale = np.ones(n_classes, dtype=np.float64)
    for c in range(n_classes):
        vals = train_residual[pred_train == c]
        if vals.size:
            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
            class_center[c] = med
            class_scale[c] = max(1.4826 * mad, 1e-6)

    eval_residual = (
        ((x_eval - means[pred_eval]) ** 2) / variances[pred_eval]
    ).mean(axis=1)
    scores = (eval_residual - class_center[pred_eval]) / class_scale[pred_eval]
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)

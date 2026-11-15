import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Class-specific shrunk diagonal variances model intent-dependent feature tolerances that a tied covariance cannot capture. "
    "A query is suspicious when its classifier-selected class assigns it an unusually large heteroscedastic residual relative to that class's training examples."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0041"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]

    global_mean = np.mean(x_train, axis=0)
    global_var = np.var(x_train, axis=0)
    variance_floor = max(float(np.mean(global_var)) * 1e-6, 1e-8)
    global_var = np.maximum(global_var, variance_floor)

    means = np.empty((n_classes, dim), dtype=float)
    variances = np.empty((n_classes, dim), dtype=float)
    shrinkage = 24.0

    for cls in range(n_classes):
        members = x_train[labels == cls]
        count = members.shape[0]
        if count == 0:
            means[cls] = global_mean
            variances[cls] = global_var
        else:
            mean = np.mean(members, axis=0)
            raw_var = np.mean((members - mean) ** 2, axis=0)
            means[cls] = mean
            variances[cls] = (
                count * raw_var + shrinkage * global_var
            ) / (count + shrinkage)

    variances = np.maximum(variances, variance_floor)
    log_dets = np.sum(np.log(variances), axis=1)

    train_residual = x_train - means[labels]
    train_nll = 0.5 * (
        np.sum((train_residual * train_residual) / variances[labels], axis=1)
        + log_dets[labels]
    )

    class_medians = np.empty(n_classes, dtype=float)
    class_scales = np.empty(n_classes, dtype=float)
    for cls in range(n_classes):
        values = train_nll[labels == cls]
        if values.size == 0:
            values = train_nll
        median = np.median(values)
        q25, q75 = np.percentile(values, [25.0, 75.0])
        class_medians[cls] = median
        class_scales[cls] = max(q75 - q25, 1e-6)

    predicted = np.argmax(ev.logits, axis=1)
    eval_residual = x_eval - means[predicted]
    eval_nll = 0.5 * (
        np.sum(
            (eval_residual * eval_residual) / variances[predicted],
            axis=1,
        )
        + log_dets[predicted]
    )

    return (eval_nll - class_medians[predicted]) / class_scales[predicted]

import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific shrunk diagonal variances model anisotropic intent clusters that a tied covariance misses, while robust per-class calibration makes density deviations comparable across intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels)
    logits = np.asarray(ev.logits, dtype=float)

    global_var = np.var(x_train, axis=0) + 1e-8
    var_floor = np.maximum(global_var * 1e-4, 1e-8)
    classes = np.unique(labels)
    dim = x_train.shape[1]

    # Restrict scoring to classifier-plausible intents, avoiding an expensive
    # evaluation against every class while retaining several alternatives.
    top_k = min(8, logits.shape[1])
    candidates = np.argpartition(logits, -top_k, axis=1)[:, -top_k:]
    best = np.full(x_eval.shape[0], np.inf, dtype=float)

    for cls in classes:
        train_mask = labels == cls
        xc = x_train[train_mask]
        n_cls = xc.shape[0]

        mean = np.mean(xc, axis=0)
        class_var = np.var(xc, axis=0)
        shrink = n_cls / (n_cls + 32.0)
        var = shrink * class_var + (1.0 - shrink) * global_var
        var = np.maximum(var, var_floor)
        inv_var = 1.0 / var
        log_det = np.sum(np.log(var))

        centered_train = xc - mean
        train_nll = 0.5 * (
            np.sum(centered_train * centered_train * inv_var, axis=1) + log_det
        )
        location = np.median(train_nll)
        q25, q75 = np.percentile(train_nll, [25.0, 75.0])
        scale = max(q75 - q25, 0.05 * dim, 1.0)

        eval_mask = np.any(candidates == cls, axis=1)
        if np.any(eval_mask):
            xe = x_eval[eval_mask] - mean
            eval_nll = 0.5 * (np.sum(xe * xe * inv_var, axis=1) + log_det)
            calibrated = (eval_nll - location) / scale
            best[eval_mask] = np.minimum(best[eval_mask], calibrated)

    return np.nan_to_num(best, nan=0.0, posinf=1e6, neginf=-1e6)

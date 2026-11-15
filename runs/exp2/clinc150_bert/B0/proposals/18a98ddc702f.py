import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Predicted-intent feature support can be anisotropic: deviations along dimensions that are "
    "stable for one intent should be more suspicious than deviations along naturally variable dimensions. "
    "A shrinkage diagonal covariance estimates this class-specific geometry without unstable full QDA matrices."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    n, d = x_train.shape
    k = tr.logits.shape[1]
    counts = np.bincount(y, minlength=k).astype(np.float64)

    global_mean = x_train.mean(axis=0)
    global_var = x_train.var(axis=0)
    var_floor = np.maximum(global_var * 1e-4, 1e-8)
    global_var = np.maximum(global_var, var_floor)

    means = np.tile(global_mean, (k, 1))
    variances = np.tile(global_var, (k, 1))

    # Shrink per-class diagonal estimates toward the global diagonal estimate.
    # This keeps rare intents well-conditioned while preserving class-specific variance.
    prior_strength = 20.0
    for c in range(k):
        mask = y == c
        nc = int(mask.sum())
        if nc == 0:
            continue
        xc = x_train[mask]
        mean_c = xc.mean(axis=0)
        centered = xc - mean_c
        var_c = np.mean(centered * centered, axis=0)
        weight = nc / (nc + prior_strength)
        means[c] = mean_c
        variances[c] = weight * var_c + (1.0 - weight) * global_var

    variances = np.maximum(variances, var_floor)
    log_variances = np.log(variances)

    # Calibrate diagonal negative log likelihood separately for each true intent
    # using only training examples, avoiding classes with inherently broader support
    # receiving systematically higher anomaly scores.
    train_nll = np.empty(n, dtype=np.float64)
    for c in range(k):
        mask = y == c
        if not np.any(mask):
            continue
        diff = x_train[mask] - means[c]
        train_nll[mask] = 0.5 * np.sum(
            diff * diff / variances[c] + log_variances[c],
            axis=1,
        )

    cal_mean = np.zeros(k, dtype=np.float64)
    cal_scale = np.ones(k, dtype=np.float64)
    for c in range(k):
        vals = train_nll[y == c]
        if vals.size:
            cal_mean[c] = vals.mean()
            cal_scale[c] = max(vals.std(), 1e-6)

    pred = np.argmax(ev.logits, axis=1)
    out = np.empty(x_eval.shape[0], dtype=np.float64)
    for c in range(k):
        mask = pred == c
        if not np.any(mask):
            continue
        diff = x_eval[mask] - means[c]
        nll = 0.5 * np.sum(
            diff * diff / variances[c] + log_variances[c],
            axis=1,
        )
        out[mask] = (nll - cal_mean[c]) / cal_scale[c]

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

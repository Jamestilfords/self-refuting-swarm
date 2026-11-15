import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Each intent has coordinates with different within-class variability, so a query predicted as that intent "
    "should not exceed many class-specific standardized residual thresholds. Aggregating soft coordinate exceedances "
    "detects broad unsupported deviations without requiring a full covariance estimate."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0094", "m_0027"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z_eval = np.asarray(ev.feats, dtype=float)
    pred = np.argmax(np.asarray(ev.logits), axis=1)

    n_classes = int(max(np.max(y), np.max(pred)) + 1)
    dim = x.shape[1]
    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0) + 1e-6

    means = np.empty((n_classes, dim), dtype=float)
    scales = np.empty((n_classes, dim), dtype=float)
    thresholds = np.empty((n_classes, dim), dtype=float)

    for c in range(n_classes):
        xc = x[y == c]
        if len(xc) == 0:
            means[c] = global_mean
            scales[c] = np.sqrt(global_var)
            thresholds[c] = 1.65
            continue

        mean_c = np.mean(xc, axis=0)
        var_c = np.var(xc, axis=0)
        shrink = len(xc) / (len(xc) + 20.0)
        var_c = shrink * var_c + (1.0 - shrink) * global_var
        scale_c = np.sqrt(np.maximum(var_c, 1e-6))

        residuals = np.abs((xc - mean_c) / scale_c)
        means[c] = mean_c
        scales[c] = scale_c
        thresholds[c] = np.quantile(residuals, 0.90, axis=0)

    out = np.empty(len(z_eval), dtype=float)
    for c in np.unique(pred):
        idx = pred == c
        residual = np.abs((z_eval[idx] - means[c]) / scales[c])
        excess = np.clip((residual - thresholds[c]) / 0.25, -40.0, 40.0)
        out[idx] = np.mean(np.logaddexp(0.0, excess), axis=1)

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

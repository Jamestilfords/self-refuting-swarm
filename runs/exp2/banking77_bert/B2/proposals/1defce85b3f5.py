import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Each intent can occupy a differently dispersed diagonal feature region, so a "
    "predicted-intent diagonal likelihood detects coordinate-wise violations hidden by tied covariance."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0102"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = x.mean(axis=0)
    global_var = np.maximum(x.var(axis=0), 1e-6)

    means = np.repeat(global_mean[None, :], k, axis=0)
    variances = np.repeat(global_var[None, :], k, axis=0)
    counts = np.bincount(y, minlength=k).astype(np.float64)

    # Shrink sparse class-wise diagonal estimates toward the global diagonal model.
    shrink_strength = 24.0
    for c in range(k):
        xc = x[y == c]
        n = xc.shape[0]
        if n == 0:
            continue
        mu = xc.mean(axis=0)
        if n > 1:
            vc = xc.var(axis=0)
        else:
            vc = global_var
        weight = n / (n + shrink_strength)
        means[c] = weight * mu + (1.0 - weight) * global_mean
        variances[c] = np.maximum(
            weight * vc + (1.0 - weight) * global_var, 1e-6
        )

    train_pred = np.argmax(tr.logits, axis=1)
    tx_mu = means[train_pred]
    tx_var = variances[train_pred]
    train_nll = 0.5 * (
        np.sum((x - tx_mu) ** 2 / tx_var + np.log(tx_var), axis=1)
        + d * np.log(2.0 * np.pi)
    )

    # Predicted-class calibration makes likelihoods comparable across intents
    # with inherently different diagonal volumes.
    centers = np.zeros(k, dtype=np.float64)
    scales = np.ones(k, dtype=np.float64)
    for c in range(k):
        values = train_nll[train_pred == c]
        if values.size:
            center = np.median(values)
            mad = np.median(np.abs(values - center))
            centers[c] = center
            scales[c] = max(1.4826 * mad, 1e-4)

    pred = np.argmax(ev.logits, axis=1)
    ez_mu = means[pred]
    ez_var = variances[pred]
    eval_nll = 0.5 * (
        np.sum((z - ez_mu) ** 2 / ez_var + np.log(ez_var), axis=1)
        + d * np.log(2.0 * np.pi)
    )
    scores = (eval_nll - centers[pred]) / scales[pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

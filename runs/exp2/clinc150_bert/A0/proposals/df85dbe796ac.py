import numpy as np
from scipy.special import logsumexp

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Different intents can occupy differently dispersed regions of representation space, so a "
    "shrinkage diagonal Gaussian mixture assigns low density to queries unsupported by every intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0083"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]

    # A compact shared projection makes class-specific diagonal variance estimation
    # stable while retaining substantially richer geometry than a tied covariance.
    projector = ctx.h.pca(tr.pooled, 64)
    x = np.asarray(projector.transform(tr.pooled), dtype=float)
    z = np.asarray(projector.transform(ev.pooled), dtype=float)
    d = x.shape[1]

    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0)
    variance_floor = max(float(np.mean(global_var)) * 1e-5, 1e-8)
    global_var = np.maximum(global_var, variance_floor)

    counts = np.bincount(y, minlength=k).astype(float)
    means = np.tile(global_mean, (k, 1))
    variances = np.tile(global_var, (k, 1))

    for c in range(k):
        xc = x[y == c]
        n_c = xc.shape[0]
        if n_c == 0:
            continue
        means[c] = np.mean(xc, axis=0)
        if n_c > 1:
            local_var = np.var(xc, axis=0)
            # Empirical-Bayes shrinkage prevents unstable variance estimates
            # for less frequent intents.
            weight = n_c / (n_c + 25.0)
            variances[c] = weight * local_var + (1.0 - weight) * global_var
        variances[c] = np.maximum(variances[c], variance_floor)

    log_prior = np.log((counts + 1.0) / (np.sum(counts) + k))
    log_components = np.empty((z.shape[0], k), dtype=float)
    constant = d * np.log(2.0 * np.pi)

    for c in range(k):
        diff = z - means[c]
        quadratic = np.sum((diff * diff) / variances[c], axis=1)
        log_det = np.sum(np.log(variances[c]))
        log_components[:, c] = log_prior[c] - 0.5 * (
            quadratic + log_det + constant
        )

    scores = -logsumexp(log_components, axis=1)

    train_components = np.empty((x.shape[0], k), dtype=float)
    for c in range(k):
        diff = x - means[c]
        quadratic = np.sum((diff * diff) / variances[c], axis=1)
        log_det = np.sum(np.log(variances[c]))
        train_components[:, c] = log_prior[c] - 0.5 * (
            quadratic + log_det + constant
        )
    train_scores = -logsumexp(train_components, axis=1)

    center = float(np.median(train_scores))
    scale = float(np.median(np.abs(train_scores - center))) * 1.4826
    scale = max(scale, 1e-6)
    return np.nan_to_num((scores - center) / scale, nan=0.0, posinf=1e6, neginf=-1e6)

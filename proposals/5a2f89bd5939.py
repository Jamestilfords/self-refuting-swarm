import numpy as np
from scipy.special import logsumexp

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A classifier-conditioned mixture of per-class shrunk diagonal Gaussian densities penalizes "
    "queries whose pooled representation is implausible under the intents receiving logit support."
)
SOURCE = "https://cdn.amazon.science/64/48/4d1e599c4472859e27a9f7647786/few-shot-out-of-domain-intent-detection-with-covariance-corrected-mahalanobis-distance.pdf"
PARENTS = []
MEMORY_USED = ["m_0002", "m_0013", "m_0027"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = x.mean(axis=0)
    global_var = x.var(axis=0)
    floor = max(float(np.median(global_var)) * 1e-4, 1e-8)
    global_var = np.maximum(global_var, floor)

    means = np.empty((k, d), dtype=np.float64)
    variances = np.empty((k, d), dtype=np.float64)
    counts = np.bincount(y, minlength=k).astype(np.float64)

    for c in range(k):
        xc = x[y == c]
        n = xc.shape[0]
        if n == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = xc.mean(axis=0)
            class_var = xc.var(axis=0) if n > 1 else global_var
            shrink = n / (n + 20.0)
            variances[c] = np.maximum(
                shrink * class_var + (1.0 - shrink) * global_var,
                floor,
            )

    logits = np.asarray(ev.logits, dtype=np.float64)
    log_post = logits - logsumexp(logits, axis=1, keepdims=True)

    mixture_terms = np.empty((z.shape[0], k), dtype=np.float64)
    const = d * np.log(2.0 * np.pi)
    for c in range(k):
        diff = z - means[c]
        quad = np.sum((diff * diff) / variances[c], axis=1)
        log_det = np.sum(np.log(variances[c]))
        log_density = -0.5 * (quad + log_det + const)
        mixture_terms[:, c] = log_post[:, c] + log_density

    scores = -logsumexp(mixture_terms, axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)

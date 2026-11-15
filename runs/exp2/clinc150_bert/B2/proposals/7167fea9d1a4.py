import numpy as np
from scipy.special import logsumexp

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Class-specific shrunk diagonal covariances model intent-dependent feature variability, while "
    "disagreement between classifier probabilities and generative class posteriors exposes forced intent assignments."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0074"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    n, d = x_train.shape
    k = tr.logits.shape[1]

    global_mean = np.mean(x_train, axis=0)
    global_var = np.mean((x_train - global_mean) ** 2, axis=0)
    global_var = np.maximum(global_var, 1e-6)

    means = np.empty((k, d), dtype=np.float64)
    variances = np.empty((k, d), dtype=np.float64)
    log_priors = np.empty(k, dtype=np.float64)

    counts = np.bincount(y, minlength=k).astype(np.float64)
    for c in range(k):
        mask = y == c
        nc = int(counts[c])
        if nc > 0:
            xc = x_train[mask]
            mu = np.mean(xc, axis=0)
            raw_var = np.mean((xc - mu) ** 2, axis=0)
            shrink = nc / (nc + 32.0)
            var = shrink * raw_var + (1.0 - shrink) * global_var
            means[c] = mu
            variances[c] = np.maximum(var, 1e-6)
        else:
            means[c] = global_mean
            variances[c] = global_var
        log_priors[c] = np.log((counts[c] + 1.0) / (n + k))

    log_det = np.sum(np.log(variances), axis=1)
    constant = d * np.log(2.0 * np.pi)

    def generative_terms(x, logits):
        m = x.shape[0]
        log_joint = np.empty((m, k), dtype=np.float64)
        for c in range(k):
            delta = x - means[c]
            mahal = np.sum((delta * delta) / variances[c], axis=1)
            log_joint[:, c] = -0.5 * (mahal + log_det[c] + constant) + log_priors[c]

        log_mix = logsumexp(log_joint, axis=1)
        log_q = log_joint - log_mix[:, None]

        p = np.asarray(ctx.h.softmax(logits), dtype=np.float64)
        p = np.clip(p, 1e-12, 1.0)
        kl = np.sum(p * (np.log(p) - log_q), axis=1)
        return -log_mix, kl

    train_nll, train_kl = generative_terms(x_train, tr.logits)
    eval_nll, eval_kl = generative_terms(x_eval, ev.logits)

    nll_mean = np.mean(train_nll)
    kl_mean = np.mean(train_kl)
    nll_scale = max(np.std(train_nll), 1e-6)
    kl_scale = max(np.std(train_kl), 1e-6)

    return (eval_nll - nll_mean) / nll_scale + (eval_kl - kl_mean) / kl_scale

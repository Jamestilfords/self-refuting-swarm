import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "A query's classifier distribution should place probability on intents whose "
    "class-specific diagonal feature models support it. The gap between soft logit-weighted "
    "feature surprise and the best-supported intent exposes logit-feature disagreement."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0094", "m_0027"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x_train.shape[1]

    global_mean = np.mean(x_train, axis=0)
    global_var = np.var(x_train, axis=0) + 1e-6

    means = np.empty((k, d), dtype=np.float64)
    variances = np.empty((k, d), dtype=np.float64)
    priors = np.empty(k, dtype=np.float64)

    for c in range(k):
        xc = x_train[y == c]
        n_c = len(xc)
        if n_c == 0:
            means[c] = global_mean
            variances[c] = global_var
            priors[c] = 1.0 / max(k, 1)
            continue

        means[c] = np.mean(xc, axis=0)
        raw_var = np.var(xc, axis=0) if n_c > 1 else global_var
        shrink = d / (d + n_c)
        variances[c] = (1.0 - shrink) * raw_var + shrink * global_var
        variances[c] = np.maximum(variances[c], 1e-6)
        priors[c] = n_c / len(x_train)

    log_densities = np.empty((len(x_eval), k), dtype=np.float64)
    log_norm_const = d * np.log(2.0 * np.pi)

    for c in range(k):
        diff = x_eval - means[c]
        mahal = np.sum((diff * diff) / variances[c], axis=1)
        log_det = np.sum(np.log(variances[c]))
        log_densities[:, c] = -0.5 * (mahal + log_det + log_norm_const)

    logits = np.asarray(ev.logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=1, keepdims=True)

    feature_surprise = -np.sum(probs * log_densities, axis=1)
    best_surprise = -np.max(log_densities, axis=1)

    log_priors = np.log(np.maximum(priors, 1e-12))
    posterior_support = log_densities + log_priors[None, :]
    best_posterior_surprise = -np.max(posterior_support, axis=1)

    score = (feature_surprise - best_surprise) + 0.25 * best_posterior_surprise
    return np.nan_to_num(score, nan=0.0, posinf=1e12, neginf=-1e12)

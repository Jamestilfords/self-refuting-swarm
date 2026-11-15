import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "Class-conditional deviations in the centered log Dirichlet concentration profile detect queries whose evidential shape is unlike the in-scope class they are predicted as. This uses both relative evidence allocation and total concentration, so it is not a monotone transform of energy or vacuity."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8
    k = tr.logits.shape[1]

    a_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), eps)

    def profile(alpha):
        log_alpha = np.log(alpha)
        clr = log_alpha - np.mean(log_alpha, axis=1, keepdims=True)
        strength = np.log(np.sum(alpha, axis=1, keepdims=True))
        return np.concatenate([clr, strength], axis=1)

    z_tr = profile(a_tr)
    z_ev = profile(a_ev)
    d = z_tr.shape[1]

    global_mu = np.mean(z_tr, axis=0)
    global_var = np.mean((z_tr - global_mu) ** 2, axis=0) + 1e-4

    means = np.empty((k, d), dtype=float)
    variances = np.empty((k, d), dtype=float)
    prior_weight = 20.0

    for c in range(k):
        mask = tr.labels == c
        n = int(np.sum(mask))
        if n == 0:
            means[c] = global_mu
            variances[c] = global_var
        else:
            z_c = z_tr[mask]
            mu_c = np.mean(z_c, axis=0)
            means[c] = (n * mu_c + prior_weight * global_mu) / (n + prior_weight)
            within = np.mean((z_c - mu_c) ** 2, axis=0)
            variances[c] = (
                n * within + prior_weight * global_var
            ) / (n + prior_weight)
            variances[c] = np.maximum(variances[c], 1e-4)

    predicted = np.argmax(ev.logits, axis=1)
    residual = z_ev - means[predicted]
    return np.mean((residual * residual) / variances[predicted], axis=1)

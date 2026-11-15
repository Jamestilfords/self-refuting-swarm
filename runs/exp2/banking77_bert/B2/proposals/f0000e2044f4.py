import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Class-conditional diagonal covariances retain intent-specific feature variances that a tied "
    "Gaussian discards. Shrinking each variance estimate toward the global variance stabilizes "
    "small classes, while predicted-class calibration makes anomaly scales comparable across intents."
)
SOURCE = "https://arxiv.org/abs/2106.09022"
PARENTS = []
MEMORY_USED = ["m_0128"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)

    n, d = x.shape
    k = tr.logits.shape[1]
    counts = np.bincount(y, minlength=k).astype(np.float64)

    global_mean = np.mean(x, axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    global_var = np.maximum(global_var, 1e-8)

    means = np.zeros((k, d), dtype=np.float64)
    np.add.at(means, y, x)
    present = counts > 0
    means[present] /= counts[present, None]
    means[~present] = global_mean

    variances = np.empty((k, d), dtype=np.float64)
    cal_mean = np.zeros(k, dtype=np.float64)
    cal_std = np.ones(k, dtype=np.float64)

    shrink_strength = 32.0
    for c in range(k):
        idx = y == c
        nc = int(np.sum(idx))
        if nc == 0:
            variances[c] = global_var
            continue

        diff = x[idx] - means[c]
        empirical_var = np.mean(diff * diff, axis=0)
        variances[c] = (
            nc * empirical_var + shrink_strength * global_var
        ) / (nc + shrink_strength)
        variances[c] = np.maximum(variances[c], 1e-8)

        train_nll = 0.5 * (
            np.sum((diff * diff) / variances[c], axis=1)
            + np.sum(np.log(variances[c]))
        )
        cal_mean[c] = np.mean(train_nll)
        cal_std[c] = max(np.std(train_nll), 1e-6)

    pred = np.argmax(ev.logits, axis=1)
    chosen_mean = means[pred]
    chosen_var = variances[pred]
    diff = z - chosen_mean

    nll = 0.5 * (
        np.sum((diff * diff) / chosen_var, axis=1)
        + np.sum(np.log(chosen_var), axis=1)
    )
    return (nll - cal_mean[pred]) / cal_std[pred]

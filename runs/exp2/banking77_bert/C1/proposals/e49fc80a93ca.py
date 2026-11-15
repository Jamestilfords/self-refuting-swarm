import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query should resemble the feature distribution of the intent selected by the classifier, not merely some nearby intent. Predicted-class diagonal Gaussian likelihood with variance shrinkage captures intent-specific spread while remaining stable for smaller classes."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    k = tr.logits.shape[1]
    d = x.shape[1]
    global_mean = np.mean(x, axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    floor = max(float(np.mean(global_var)) * 1e-6, 1e-10)
    global_var = np.maximum(global_var, floor)

    means = np.tile(global_mean, (k, 1))
    variances = np.tile(global_var, (k, 1))
    counts = np.zeros(k, dtype=float)

    for c in range(k):
        xc = x[y == c]
        n_c = xc.shape[0]
        counts[c] = n_c
        if n_c > 0:
            mean_c = np.mean(xc, axis=0)
            means[c] = mean_c
            if n_c > 1:
                var_c = np.mean((xc - mean_c) ** 2, axis=0)
            else:
                var_c = global_var
            weight = n_c / (n_c + 20.0)
            variances[c] = np.maximum(
                weight * var_c + (1.0 - weight) * global_var,
                floor,
            )

    pred = np.argmax(ev.logits, axis=1)
    diff = z - means[pred]
    var = variances[pred]
    mahal = np.sum((diff * diff) / var, axis=1)
    logdet = np.sum(np.log(var), axis=1)

    prior = (counts[pred] + 1.0) / (np.sum(counts) + float(k))
    nll = 0.5 * (mahal + logdet) - np.log(prior)

    tie_break = 1e-12 * np.sum(z * np.arange(1, d + 1, dtype=float), axis=1)
    return np.nan_to_num(nll + tie_break, nan=0.0, posinf=1e30, neginf=-1e30)

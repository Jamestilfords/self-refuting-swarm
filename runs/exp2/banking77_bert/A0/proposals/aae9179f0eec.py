import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Per-intent diagonal variances expose representation coordinates that violate the "
    "predicted intent's individual tolerance profile. Counting and sizing coordinatewise "
    "tail exceedances detects distributed mismatch that a single pooled Mahalanobis score can dilute."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0083"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)
    n, d = x_train.shape
    k = tr.logits.shape[1]

    global_mean = np.mean(x_train, axis=0)
    global_var = np.mean((x_train - global_mean) ** 2, axis=0)
    global_var = np.maximum(global_var, 1e-8)

    counts = np.bincount(labels, minlength=k).astype(float)
    means = np.tile(global_mean, (k, 1))
    variances = np.tile(global_var, (k, 1))
    fences = np.full((k, d), 1.645, dtype=float)

    for c in range(k):
        idx = labels == c
        nc = int(np.sum(idx))
        if nc == 0:
            continue

        xc = x_train[idx]
        mu = np.mean(xc, axis=0)
        raw_var = np.mean((xc - mu) ** 2, axis=0)

        # Small intents borrow variance information from the full in-scope population.
        weight = nc / (nc + 20.0)
        var = weight * raw_var + (1.0 - weight) * global_var
        var = np.maximum(var, 1e-8)

        means[c] = mu
        variances[c] = var

        z_train = np.abs(xc - mu) / np.sqrt(var)
        empirical_fence = np.quantile(z_train, 0.90, axis=0)

        # Avoid unstable per-coordinate quantiles for rare intents.
        fence_weight = nc / (nc + 10.0)
        fences[c] = fence_weight * empirical_fence + (1.0 - fence_weight) * 1.645

    predicted = np.argmax(ev.logits, axis=1)
    mu_eval = means[predicted]
    var_eval = variances[predicted]
    fence_eval = np.maximum(fences[predicted], 0.25)

    standardized = np.abs(x_eval - mu_eval) / np.sqrt(var_eval)
    excess = np.maximum(standardized / fence_eval - 1.0, 0.0)

    # A small continuous residual term prevents broad ties when no coordinate crosses its fence.
    return np.mean(excess, axis=1) + 0.01 * np.mean(standardized, axis=1)

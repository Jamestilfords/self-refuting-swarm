import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query should resemble the feature distribution of the intent predicted by the classifier, not merely be close to some class. Per-class diagonal variances retain intent-specific uncertainty while shrinkage stabilizes rare classes."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    n_classes = tr.logits.shape[1]
    dim = x.shape[1]
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)

    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0)
    global_var = np.maximum(global_var, 1e-6)

    sums = np.zeros((n_classes, dim), dtype=np.float64)
    sums_sq = np.zeros((n_classes, dim), dtype=np.float64)
    np.add.at(sums, y, x)
    np.add.at(sums_sq, y, x * x)

    safe_counts = np.maximum(counts, 1.0)[:, None]
    means = sums / safe_counts
    means[counts == 0] = global_mean

    empirical_var = sums_sq / safe_counts - means * means
    empirical_var = np.maximum(empirical_var, 0.0)

    # Rare intents borrow featurewise variance from the global population,
    # whereas well-represented intents keep their own uncertainty profile.
    weight = (counts / (counts + 32.0))[:, None]
    variances = weight * empirical_var + (1.0 - weight) * global_var[None, :]
    variances = np.maximum(variances, 1e-6)

    predicted = np.argmax(ev.logits, axis=1)
    chosen_means = means[predicted]
    chosen_vars = variances[predicted]

    residual = z - chosen_means
    mahalanobis = np.sum((residual * residual) / chosen_vars, axis=1)
    logdet = np.sum(np.log(chosen_vars), axis=1)

    # Negative class-conditional diagonal Gaussian log likelihood, omitting
    # the shared dimensional normalizing constant.
    return 0.5 * (mahalanobis + logdet)

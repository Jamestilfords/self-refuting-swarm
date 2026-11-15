import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Per-intent diagonal variance identifies coordinates where an intent permits little variation. "
    "Averaging the largest class-normalized coordinate residuals detects sparse off-manifold departures that a summed Mahalanobis score can dilute."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0053", "m_0073"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    n, d = x.shape
    k = tr.logits.shape[1]
    eps = 1e-8
    shrinkage = 24.0
    top_dims = min(24, d)

    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0) + eps
    means = np.tile(global_mean, (k, 1))
    variances = np.tile(global_var, (k, 1))

    for c in range(k):
        xc = x[y == c]
        nc = len(xc)
        if nc == 0:
            continue
        mean_c = np.mean(xc, axis=0)
        var_c = np.mean((xc - mean_c) ** 2, axis=0)
        means[c] = mean_c
        variances[c] = (
            nc * var_c + shrinkage * global_var
        ) / (nc + shrinkage)

    variances = np.maximum(variances, eps)
    pred = np.argmax(ev.logits, axis=1)

    residual_sq = (z - means[pred]) ** 2 / variances[pred]
    largest = np.partition(residual_sq, d - top_dims, axis=1)[:, -top_dims:]
    score_values = np.mean(largest, axis=1)

    return np.nan_to_num(
        score_values,
        nan=0.0,
        posinf=1e12,
        neginf=-1e12,
    )

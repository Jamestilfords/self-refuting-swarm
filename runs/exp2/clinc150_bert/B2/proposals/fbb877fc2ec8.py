import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Per-intent diagonal covariances preserve feature dimensions whose variability is specific to an intent, "
    "while shrinkage toward global variance prevents unstable estimates for small classes. OOS queries should "
    "have a larger nearest class-conditional standardized residual."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels)

    global_var = np.var(x_train, axis=0)
    scale_floor = max(float(np.median(global_var)) * 1e-4, 1e-8)
    global_var = np.maximum(global_var, scale_floor)

    best = np.full(x_eval.shape[0], np.inf, dtype=np.float64)
    for c in np.unique(labels):
        xc = x_train[labels == c]
        n_c = xc.shape[0]
        mean_c = np.mean(xc, axis=0)
        var_c = np.var(xc, axis=0)

        shrink = n_c / (n_c + 20.0)
        var_c = shrink * var_c + (1.0 - shrink) * global_var
        var_c = np.maximum(var_c, scale_floor)

        diff = x_eval - mean_c
        dist = np.sum((diff * diff) / var_c, axis=1)
        best = np.minimum(best, dist)

    best = np.minimum(best, 1e12)
    tie_break = 1e-8 * ctx.h.energy(ev.logits)
    return best + tie_break

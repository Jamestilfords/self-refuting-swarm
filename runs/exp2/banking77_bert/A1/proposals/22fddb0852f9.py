import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Intent-specific diagonal variances preserve feature dimensions that are tightly constrained for one intent "
    "but variable for another. Shrinking each class variance toward the global variance stabilizes sparse intents "
    "while making OOS queries have high minimum class-conditional Gaussian surprise."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0021"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    labels = np.asarray(tr.labels)

    global_var = np.var(x_train, axis=0) + 1e-6
    classes = np.unique(labels)
    best = np.full(x_eval.shape[0], np.inf, dtype=np.float64)

    for c in classes:
        xc = x_train[labels == c]
        n_c = xc.shape[0]
        mean_c = np.mean(xc, axis=0)

        if n_c > 1:
            var_c = np.var(xc, axis=0)
        else:
            var_c = global_var

        shrink = n_c / (n_c + 20.0)
        var_c = shrink * var_c + (1.0 - shrink) * global_var + 1e-6
        log_det = np.sum(np.log(var_c))

        diff = x_eval - mean_c
        nll = 0.5 * (np.sum((diff * diff) / var_c, axis=1) + log_det)
        best = np.minimum(best, nll)

    return np.nan_to_num(best, nan=0.0, posinf=1e12, neginf=-1e12)

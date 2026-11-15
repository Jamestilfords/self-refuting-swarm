import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Class-specific shrunk diagonal Gaussian likelihoods retain intent-dependent feature "
    "scales while avoiding unstable full covariance estimates. OOS queries should have "
    "low likelihood under every in-scope intent, rather than only under their predicted intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0021"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)

    classes, counts = np.unique(labels, return_counts=True)
    global_var = np.var(x_train, axis=0) + 1e-6
    best_nll = np.full(x_eval.shape[0], np.inf, dtype=np.float64)

    for c, n_c in zip(classes, counts):
        members = x_train[labels == c]
        mean = np.mean(members, axis=0)
        if n_c > 1:
            class_var = np.var(members, axis=0)
        else:
            class_var = global_var

        shrink = float(n_c) / float(n_c + 32)
        var = shrink * class_var + (1.0 - shrink) * global_var + 1e-6
        diff = x_eval - mean
        nll = 0.5 * (
            np.sum((diff * diff) / var, axis=1) + np.sum(np.log(var))
        )
        best_nll = np.minimum(best_nll, nll)

    return np.nan_to_num(best_nll, nan=0.0, posinf=1e12, neginf=-1e12)

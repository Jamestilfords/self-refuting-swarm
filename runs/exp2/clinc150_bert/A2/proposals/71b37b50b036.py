import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Intent classes can have substantially different feature dispersions, so classifier-selected diagonal Gaussian surprise with sample-size shrinkage can expose queries that are implausible for their predicted intent. Per-class empirical calibration removes density offsets caused by differing class volumes."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0021"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_var = np.var(x, axis=0) + 1e-6
    means = np.zeros((k, d), dtype=float)
    variances = np.zeros((k, d), dtype=float)
    counts = np.bincount(y, minlength=k).astype(float)
    shrink = 20.0

    for c in range(k):
        xc = x[y == c]
        if xc.shape[0] == 0:
            means[c] = np.mean(x, axis=0)
            variances[c] = global_var
        else:
            means[c] = np.mean(xc, axis=0)
            local_var = np.var(xc, axis=0)
            variances[c] = (
                xc.shape[0] * local_var + shrink * global_var
            ) / (xc.shape[0] + shrink)
            variances[c] = np.maximum(variances[c], 1e-6)

    def nll_selected(a, classes):
        mu = means[classes]
        va = variances[classes]
        diff = a - mu
        return 0.5 * (
            np.sum(diff * diff / va, axis=1)
            + np.sum(np.log(va), axis=1)
            + d * np.log(2.0 * np.pi)
        )

    train_nll = nll_selected(x, y)
    pred = np.argmax(ev.logits, axis=1).astype(int)
    eval_nll = nll_selected(z, pred)

    tail = np.empty(z.shape[0], dtype=float)
    for c in range(k):
        mask = pred == c
        reference = np.sort(train_nll[y == c])
        if reference.size == 0:
            reference = np.sort(train_nll)
        rank = np.searchsorted(reference, eval_nll[mask], side="right")
        tail[mask] = rank / (reference.size + 1.0)

    return tail + 1e-8 * eval_nll

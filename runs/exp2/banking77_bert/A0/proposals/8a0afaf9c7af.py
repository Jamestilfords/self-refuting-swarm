import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A shrunk diagonal covariance gives each intent its own coordinatewise tolerance while "
    "borrowing strength from global feature variability. Queries with unusually low likelihood "
    "under their logit-predicted intent receive high train-calibrated OOS scores."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0105"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = np.asarray(tr.feats, dtype=float)
    xev = np.asarray(ev.feats, dtype=float)
    ytr = np.asarray(tr.labels, dtype=int)

    n_classes = tr.logits.shape[1]
    pred_ev = np.argmax(ev.logits, axis=1)
    d = xtr.shape[1]

    global_mean = np.mean(xtr, axis=0)
    global_var = np.var(xtr, axis=0) + 1e-6

    means = np.empty((n_classes, d), dtype=float)
    variances = np.empty((n_classes, d), dtype=float)
    train_nll = [None] * n_classes

    for c in range(n_classes):
        xc = xtr[ytr == c]
        count = xc.shape[0]

        if count == 0:
            mean_c = global_mean
            var_c = global_var
        else:
            mean_c = np.mean(xc, axis=0)
            raw_var = np.var(xc, axis=0) + 1e-6
            shrink = count / (count + 50.0)
            var_c = shrink * raw_var + (1.0 - shrink) * global_var

        means[c] = mean_c
        variances[c] = var_c

        if count > 0:
            z = (xc - mean_c) ** 2 / var_c
            train_nll[c] = 0.5 * np.sum(z + np.log(var_c), axis=1)
        else:
            train_nll[c] = np.array([0.0])

    out = np.empty(xev.shape[0], dtype=float)
    for c in range(n_classes):
        mask = pred_ev == c
        if not np.any(mask):
            continue

        xe = xev[mask]
        z = (xe - means[c]) ** 2 / variances[c]
        nll = 0.5 * np.sum(z + np.log(variances[c]), axis=1)

        reference = np.sort(train_nll[c])
        percentile = np.searchsorted(reference, nll, side="right") / float(reference.size)

        scale = np.std(reference) + 1e-6
        out[mask] = percentile + 1e-4 * (nll - np.mean(reference)) / scale

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

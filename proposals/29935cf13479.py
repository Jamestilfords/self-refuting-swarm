import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Shrunk per-intent diagonal variances make residuals comparable across coordinates, while "
    "the distribution of standardized residual energy across fixed feature blocks captures atypical "
    "heteroscedastic deviation patterns beyond a single Mahalanobis radius."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0094"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    xe = np.asarray(ev.feats, dtype=np.float64)

    n, d = x.shape
    k = tr.logits.shape[1]
    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0) + 1e-6

    means = np.empty((k, d), dtype=np.float64)
    variances = np.empty((k, d), dtype=np.float64)
    counts = np.bincount(y, minlength=k).astype(np.float64)

    for c in range(k):
        xc = x[y == c]
        nc = xc.shape[0]
        if nc == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = np.mean(xc, axis=0)
            class_var = np.var(xc, axis=0) + 1e-6
            shrink = nc / (nc + 32.0)
            variances[c] = shrink * class_var + (1.0 - shrink) * global_var

    blocks = 16
    block_id = np.arange(d) % blocks

    def residual_statistics(z, pred):
        mu = means[pred]
        va = variances[pred]
        standardized_sq = (z - mu) ** 2 / va
        radius = np.mean(standardized_sq, axis=1)

        block_energy = np.empty((z.shape[0], blocks), dtype=np.float64)
        for b in range(blocks):
            block_energy[:, b] = np.sum(standardized_sq[:, block_id == b], axis=1)

        total = np.sum(block_energy, axis=1, keepdims=True)
        fractions = block_energy / np.maximum(total, 1e-10)
        return radius, fractions

    rad_tr, frac_tr = residual_statistics(x, pred_tr)
    rad_ev, frac_ev = residual_statistics(xe, pred_ev)

    global_frac_mean = np.mean(frac_tr, axis=0)
    global_frac_var = np.var(frac_tr, axis=0) + 1e-6
    frac_mean = np.empty((k, blocks), dtype=np.float64)
    frac_var = np.empty((k, blocks), dtype=np.float64)

    for c in range(k):
        fc = frac_tr[pred_tr == c]
        nc = fc.shape[0]
        if nc < 4:
            frac_mean[c] = global_frac_mean
            frac_var[c] = global_frac_var
        else:
            local_mean = np.mean(fc, axis=0)
            local_var = np.var(fc, axis=0) + 1e-6
            shrink = nc / (nc + 32.0)
            frac_mean[c] = shrink * local_mean + (1.0 - shrink) * global_frac_mean
            frac_var[c] = shrink * local_var + (1.0 - shrink) * global_frac_var

    def balance_stat(frac, pred):
        delta = frac - frac_mean[pred]
        return np.mean(delta ** 2 / frac_var[pred], axis=1)

    bal_tr = balance_stat(frac_tr, pred_tr)
    bal_ev = balance_stat(frac_ev, pred_ev)

    global_rad_sorted = np.sort(rad_tr)
    global_bal_sorted = np.sort(bal_tr)
    scores = np.empty(xe.shape[0], dtype=np.float64)

    for c in range(k):
        test_mask = pred_ev == c
        if not np.any(test_mask):
            continue

        train_mask = pred_tr == c
        if np.sum(train_mask) >= 12:
            sr = np.sort(rad_tr[train_mask])
            sb = np.sort(bal_tr[train_mask])
        else:
            sr = global_rad_sorted
            sb = global_bal_sorted

        r_tail = np.searchsorted(sr, rad_ev[test_mask], side="right") / float(sr.size)
        b_tail = np.searchsorted(sb, bal_ev[test_mask], side="right") / float(sb.size)
        scores[test_mask] = r_tail + b_tail + 1e-6 * rad_ev[test_mask]

    return np.nan_to_num(scores, nan=0.0, posinf=2.0, neginf=0.0)

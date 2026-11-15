import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Within-class trailing covariance directions are tightly constrained for in-scope intents. A class-conditional higher-criticism scan detects an unusual excess of coordinate-wise tail violations in that low-variance subspace without reducing the signal to a single Mahalanobis radius."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0017", "m_0087"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d_trail = min(48, x_train.shape[1])

    counts = np.bincount(y, minlength=k).astype(np.float64)
    sums = np.zeros((k, x_train.shape[1]), dtype=np.float64)
    np.add.at(sums, y, x_train)
    global_mean = x_train.mean(axis=0)
    means = np.empty_like(sums)
    valid = counts > 0
    means[valid] = sums[valid] / counts[valid, None]
    means[~valid] = global_mean

    residuals = x_train - means[y]
    cov = residuals.T @ residuals / max(len(x_train) - 1, 1)
    cov.flat[:: cov.shape[0] + 1] += 1e-6
    eigvals, eigvecs = np.linalg.eigh(cov)
    basis = eigvecs[:, :d_trail]

    z_train = residuals @ basis
    scales = np.ones((k, d_trail), dtype=np.float64)
    for c in range(k):
        members = z_train[y == c]
        if len(members) >= 2:
            scales[c] = np.sqrt(np.mean(members * members, axis=0))
    global_scale = np.sqrt(np.mean(z_train * z_train, axis=0))
    scales[~valid] = global_scale
    floor = np.maximum(np.median(global_scale) * 1e-3, 1e-8)
    scales = np.maximum(scales, floor)

    sorted_abs = []
    for c in range(k):
        members = np.abs(z_train[y == c] / scales[c])
        if len(members) == 0:
            members = np.abs(z_train / scales[c])
        sorted_abs.append(np.sort(members, axis=0))

    def raw_scan(values, classes):
        out = np.empty(len(values), dtype=np.float64)
        for c in np.unique(classes):
            take = np.flatnonzero(classes == c)
            z = np.abs((values[take] - means[c]) @ basis / scales[c])
            ref = sorted_abs[c]
            n_ref = len(ref)
            pvals = np.empty_like(z)
            for j in range(d_trail):
                ranks = np.searchsorted(ref[:, j], z[:, j], side="left")
                pvals[:, j] = (n_ref - ranks + 1.0) / (n_ref + 1.0)

            pvals.sort(axis=1)
            order = np.arange(1.0, d_trail + 1.0)[None, :] / d_trail
            denom = np.sqrt(np.maximum(pvals * (1.0 - pvals), 1e-8))
            hc = (order - pvals) / denom
            hc = np.maximum(hc, 0.0)
            out[take] = np.max(hc, axis=1) + 0.05 * np.mean(-np.log(pvals), axis=1)
        return out

    train_raw = raw_scan(x_train, y)
    raw_by_class = []
    for c in range(k):
        vals = train_raw[y == c]
        if len(vals) == 0:
            vals = train_raw
        raw_by_class.append(np.sort(vals))

    pred = np.argmax(ev.logits, axis=1).astype(np.int64)
    pred = np.clip(pred, 0, k - 1)
    eval_raw = raw_scan(x_eval, pred)

    scores = np.empty(len(x_eval), dtype=np.float64)
    for c in np.unique(pred):
        take = np.flatnonzero(pred == c)
        ref = raw_by_class[c]
        ranks = np.searchsorted(ref, eval_raw[take], side="left")
        tail = (len(ref) - ranks + 1.0) / (len(ref) + 1.0)
        scores[take] = -np.log(tail) + 1e-3 * eval_raw[take]

    return np.nan_to_num(scores, nan=0.0, posinf=50.0, neginf=0.0)

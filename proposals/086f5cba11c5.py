import numpy as np
from sklearn.utils.extmath import randomized_svd

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "After removing the class-mean discriminative subspace, each known intent retains "
    "a low-dimensional residual variation pattern. Queries whose residual cannot be "
    "reconstructed by the classifier-selected intent's residual subspace are likely OOS."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0014"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = np.mean(x, axis=0)
    means = np.tile(global_mean, (k, 1))
    counts = np.bincount(y, minlength=k)
    for c in range(k):
        if counts[c] > 0:
            means[c] = np.mean(x[y == c], axis=0)

    active = np.flatnonzero(counts > 0)
    mean_matrix = means[active] - global_mean
    if mean_matrix.shape[0] > 1:
        _, s, vt = np.linalg.svd(mean_matrix, full_matrices=False)
        rank = min(32, int(np.sum(s > 1e-10)))
        u = vt[:rank].T if rank > 0 else np.empty((d, 0))
    else:
        u = np.empty((d, 0))

    def residual(a, center):
        r = a - center
        if u.shape[1]:
            r = r - (r @ u) @ u.T
        return r

    global_r = residual(x, global_mean)
    global_rank = min(8, max(1, min(global_r.shape) - 1))
    _, _, global_vt = randomized_svd(
        global_r, n_components=global_rank, n_iter=3, random_state=2718
    )
    global_err = np.sum(
        (global_r - (global_r @ global_vt.T) @ global_vt) ** 2, axis=1
    )
    global_med = np.median(global_err)
    global_scale = np.quantile(global_err, 0.9) - global_med
    global_scale = max(float(global_scale), 1e-8)

    bases = [global_vt for _ in range(k)]
    medians = np.full(k, global_med, dtype=float)
    scales = np.full(k, global_scale, dtype=float)

    for c in active:
        xc = x[y == c]
        rc = residual(xc, means[c])
        if rc.shape[0] < 4:
            continue
        rnk = min(8, rc.shape[0] - 1, rc.shape[1])
        _, _, vt = randomized_svd(rc, n_components=rnk, n_iter=3, random_state=2718 + c)
        err = np.sum((rc - (rc @ vt.T) @ vt) ** 2, axis=1)
        med = np.median(err)
        scale = np.quantile(err, 0.9) - med
        bases[c] = vt
        medians[c] = med
        scales[c] = max(float(scale), 1e-8)

    pred = np.argmax(ev.logits, axis=1)
    out = np.empty(z.shape[0], dtype=float)
    for c in np.unique(pred):
        idx = np.flatnonzero(pred == c)
        rz = residual(z[idx], means[c])
        vt = bases[c]
        err = np.sum((rz - (rz @ vt.T) @ vt) ** 2, axis=1)
        out[idx] = (err - medians[c]) / scales[c]

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

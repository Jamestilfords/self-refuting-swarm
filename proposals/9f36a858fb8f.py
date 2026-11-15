import numpy as np
from sklearn.utils.extmath import randomized_svd

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "After removing the subspace spanned by intent mean differences, in-scope examples should still lie near a shared low-dimensional nuisance manifold in the remaining coordinates. "
    "Large reconstruction error outside that residual manifold indicates semantic content unsupported by known intent variation."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0014", "m_0158", "m_0132"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels)
    z = np.asarray(ev.pooled, dtype=np.float64)

    mu = x.mean(axis=0)
    n_classes = int(y.max()) + 1
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    class_sums = np.zeros((n_classes, x.shape[1]), dtype=np.float64)
    np.add.at(class_sums, y, x)
    class_means = class_sums / np.maximum(counts[:, None], 1.0)

    mean_diffs = class_means - mu
    _, singular, vt = np.linalg.svd(mean_diffs, full_matrices=False)
    if singular.size:
        tol = singular[0] * 1e-8
        disc_rank = max(1, int(np.sum(singular > tol)))
        q_disc = vt[:disc_rank].T
    else:
        q_disc = np.zeros((x.shape[1], 0), dtype=np.float64)

    xc = x - mu
    zc = z - mu
    if q_disc.shape[1]:
        xr = xc - (xc @ q_disc) @ q_disc.T
        zr = zc - (zc @ q_disc) @ q_disc.T
    else:
        xr, zr = xc, zc

    residual_dim = xr.shape[1]
    rank = min(96, max(1, residual_dim - q_disc.shape[1]))
    _, _, vt_res = randomized_svd(
        xr,
        n_components=rank,
        n_iter=4,
        random_state=271828,
    )
    q_res = vt_res.T

    train_tail = xr - (xr @ q_res) @ q_res.T
    eval_tail = zr - (zr @ q_res) @ q_res.T
    train_error = np.sum(train_tail * train_tail, axis=1)
    eval_error = np.sum(eval_tail * eval_tail, axis=1)

    center = np.median(train_error)
    scale = np.median(np.abs(train_error - center))
    if not np.isfinite(scale) or scale < 1e-10:
        scale = np.std(train_error)
    if not np.isfinite(scale) or scale < 1e-10:
        scale = 1.0

    return np.nan_to_num(
        (eval_error - center) / scale,
        nan=0.0,
        posinf=1e12,
        neginf=-1e12,
    )

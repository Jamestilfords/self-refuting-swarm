import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A class-conditional residual score measures distance from the low-rank feature subspace of the class predicted by the classifier. This differs from global ViM residuals by detecting samples that fall off the local in-scope manifold even when they remain near the global feature distribution."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)

    n, d = x_train.shape
    k = tr.logits.shape[1]
    rng = np.random.default_rng(271828)

    global_mean = np.mean(x_train, axis=0)
    global_centered = x_train - global_mean
    global_rank = min(24, d, max(1, n - 1))
    omega = rng.standard_normal((d, min(global_rank + 8, d)))
    q, _ = np.linalg.qr(global_centered @ omega, mode="reduced")
    _, _, vt = np.linalg.svd(q.T @ global_centered, full_matrices=False)
    global_basis = vt[:global_rank]

    means = []
    bases = []
    locs = np.empty(k, dtype=np.float64)
    scales = np.empty(k, dtype=np.float64)

    for c in range(k):
        xc = x_train[labels == c]
        if xc.shape[0] < 8:
            mean = global_mean
            basis = global_basis
            centered = xc - mean
        else:
            mean = np.mean(xc, axis=0)
            centered = xc - mean
            rank = min(16, d, xc.shape[0] - 1)
            oversample = min(rank + 6, d)
            local_omega = rng.standard_normal((d, oversample))
            q_local, _ = np.linalg.qr(centered @ local_omega, mode="reduced")
            _, _, vt_local = np.linalg.svd(q_local.T @ centered, full_matrices=False)
            basis = vt_local[:rank]

        residual_sq = np.sum(centered * centered, axis=1)
        if basis.shape[0] > 0 and centered.shape[0] > 0:
            proj = centered @ basis.T
            residual_sq -= np.sum(proj * proj, axis=1)
        residual = np.sqrt(np.maximum(residual_sq, 0.0))

        if residual.size == 0:
            loc = 0.0
            scale = 1.0
        else:
            loc = float(np.median(residual))
            scale = float(np.median(np.abs(residual - loc)))
            scale = max(1.4826 * scale, 1e-6)

        means.append(mean)
        bases.append(basis)
        locs[c] = loc
        scales[c] = scale

    pred = np.argmax(ev.logits, axis=1)
    scores = np.empty(x_eval.shape[0], dtype=np.float64)

    for c in range(k):
        mask = pred == c
        if not np.any(mask):
            continue
        centered = x_eval[mask] - means[c]
        residual_sq = np.sum(centered * centered, axis=1)
        if bases[c].shape[0] > 0:
            proj = centered @ bases[c].T
            residual_sq -= np.sum(proj * proj, axis=1)
        residual = np.sqrt(np.maximum(residual_sq, 0.0))
        scores[mask] = (residual - locs[c]) / scales[c]

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

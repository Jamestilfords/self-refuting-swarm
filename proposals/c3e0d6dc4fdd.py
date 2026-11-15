import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Within-class trailing covariance directions are tightly constrained for supported intents. "
    "Classifier-selected local support in whitened trailing residual directions detects unfamiliar OOS deviations."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0006", "m_0017"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    n, d = x.shape
    k = tr.logits.shape[1]
    global_mean = np.mean(x, axis=0)

    counts = np.bincount(y, minlength=k).astype(float)
    sums = np.zeros((k, d), dtype=float)
    np.add.at(sums, y, x)
    means = np.tile(global_mean, (k, 1))
    present = counts > 0
    means[present] = sums[present] / counts[present, None]

    residuals = x - means[y]
    denom = max(n - int(np.sum(present)), 1)
    cov = (residuals.T @ residuals) / float(denom)
    scale = float(np.trace(cov) / max(d, 1))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    cov.flat[::d + 1] += 1e-5 * scale

    values, vectors = np.linalg.eigh(cov)
    q = min(32, max(8, d // 8))
    trailing_vectors = vectors[:, :q]
    trailing_inv_std = 1.0 / np.sqrt(np.maximum(values[:q], 1e-8 * scale))

    train_tail = (residuals @ trailing_vectors) * trailing_inv_std
    train_norm = np.linalg.norm(train_tail, axis=1)
    train_dirs = train_tail / np.maximum(train_norm[:, None], 1e-12)

    pred = np.argmax(np.asarray(ev.logits), axis=1)
    eval_tail = (z - means[pred]) @ trailing_vectors
    eval_tail *= trailing_inv_std
    eval_norm = np.linalg.norm(eval_tail, axis=1)
    eval_dirs = eval_tail / np.maximum(eval_norm[:, None], 1e-12)

    out = np.empty(len(z), dtype=float)
    all_train = np.arange(n)
    max_anchors = 96

    for c in range(k):
        test_idx = np.flatnonzero(pred == c)
        if test_idx.size == 0:
            continue

        train_idx = np.flatnonzero(y == c)
        if train_idx.size == 0:
            train_idx = all_train

        take = min(max_anchors, train_idx.size)
        if train_idx.size > take:
            rng = np.random.default_rng(1729 + c)
            chosen = rng.choice(train_idx, size=take, replace=False)
        else:
            chosen = train_idx

        anchors = train_dirs[chosen]
        query = eval_dirs[test_idx]
        dist_sq = (
            np.sum(query * query, axis=1, keepdims=True)
            + np.sum(anchors * anchors, axis=1)[None, :]
            - 2.0 * (query @ anchors.T)
        )
        directional = np.sqrt(np.maximum(np.min(dist_sq, axis=1), 0.0))

        class_norms = train_norm[train_idx]
        norm_scale = float(np.median(class_norms))
        if not np.isfinite(norm_scale) or norm_scale <= 1e-8:
            norm_scale = 1.0
        magnitude = np.log1p(eval_norm[test_idx] / norm_scale)

        out[test_idx] = directional + 0.1 * magnitude

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=0.0)

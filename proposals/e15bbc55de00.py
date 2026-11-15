import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions retain constrained residual structure that OOS inputs often violate. "
    "Class-selected empirical projection depth in this trailing subspace flags representations lying in sparse or extreme in-scope regions."
)
SOURCE = "https://arxiv.org/abs/2211.13527"
PARENTS = []
MEMORY_USED = ["m_0278", "m_0241", "m_0188"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.pooled, dtype=float)

    n, d = x.shape
    k = tr.logits.shape[1]
    global_mean = x.mean(axis=0)

    means = np.tile(global_mean, (k, 1))
    counts = np.bincount(y, minlength=k)
    for c in range(k):
        if counts[c] > 0:
            means[c] = x[y == c].mean(axis=0)

    centered = x - means[y]
    cov = centered.T @ centered / max(n - np.count_nonzero(counts), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    q = min(48, d)
    trail_vals = eigvals[:q]
    trail_vecs = eigvecs[:, :q]
    floor = max(float(eigvals[-1]) * 1e-6, 1e-8)
    scale = np.sqrt(np.maximum(trail_vals, floor))

    train_trail = (centered @ trail_vecs) / scale
    pred = np.argmax(ev.logits, axis=1)
    eval_trail = ((z - means[pred]) @ trail_vecs) / scale

    rng = np.random.default_rng(271828)
    directions = rng.normal(size=(q, 32))
    directions /= np.maximum(np.linalg.norm(directions, axis=0, keepdims=True), 1e-12)

    scores = np.empty(z.shape[0], dtype=float)
    global_proj = train_trail @ directions

    for c in range(k):
        idx = np.flatnonzero(pred == c)
        if idx.size == 0:
            continue

        train_idx = np.flatnonzero(y == c)
        if train_idx.size >= 4:
            ref = train_trail[train_idx] @ directions
        else:
            ref = global_proj

        query = eval_trail[idx] @ directions
        depths = np.empty((idx.size, directions.shape[1]), dtype=float)

        for j in range(directions.shape[1]):
            sorted_ref = np.sort(ref[:, j])
            left = np.searchsorted(sorted_ref, query[:, j], side="left")
            right = np.searchsorted(sorted_ref, query[:, j], side="right")
            p = (left + right + 1.0) / (2.0 * (sorted_ref.size + 1.0))
            depths[:, j] = np.minimum(p, 1.0 - p)

        scores[idx] = 1.0 - 2.0 * depths.mean(axis=1)

    return np.nan_to_num(scores, nan=1.0, posinf=1.0, neginf=0.0)

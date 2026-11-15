import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Known intents occupy characteristic central regions even in the tightly constrained "
    "low-variance within-class subspace. Random-projection Tukey depth in that trailing "
    "subspace flags samples that are extreme along any supported low-variance direction."
)
SOURCE = "https://papers.neurips.cc/paper_files/paper/2022/file/70fa5df8e3300dc30bf19bee44a56155-Paper-Conference.pdf"
PARENTS = []
MEMORY_USED = ["m_0101", "m_0017"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.feats, dtype=np.float64)

    n, dim = x.shape
    classes = tr.logits.shape[1]
    means = np.zeros((classes, dim), dtype=np.float64)
    counts = np.bincount(y, minlength=classes).astype(np.float64)

    for c in range(classes):
        mask = y == c
        if np.any(mask):
            means[c] = x[mask].mean(axis=0)
        else:
            means[c] = x.mean(axis=0)

    residuals = x - means[y]
    cov = (residuals.T @ residuals) / max(n - classes, 1)
    cov.flat[:: dim + 1] += 1e-6

    vals, vecs = np.linalg.eigh(cov)
    trailing_dim = min(32, dim)
    basis = vecs[:, :trailing_dim]
    scales = 1.0 / np.sqrt(np.maximum(vals[:trailing_dim], 1e-8))
    basis = basis * scales[None, :]

    rng = np.random.default_rng(271828)
    directions = rng.normal(size=(trailing_dim, 48))
    directions /= np.maximum(np.linalg.norm(directions, axis=0, keepdims=True), 1e-12)

    train_proj = (residuals @ basis) @ directions
    pred = np.argmax(ev.logits, axis=1)
    out = np.empty(z.shape[0], dtype=np.float64)

    global_sorted = np.sort(train_proj, axis=0)
    for c in range(classes):
        idx = np.flatnonzero(pred == c)
        if idx.size == 0:
            continue

        class_mask = y == c
        reference = train_proj[class_mask] if np.any(class_mask) else train_proj
        reference = np.sort(reference, axis=0)
        m = reference.shape[0]

        q = ((z[idx] - means[c]) @ basis) @ directions
        left = np.empty_like(q, dtype=np.float64)
        right = np.empty_like(q, dtype=np.float64)

        for j in range(q.shape[1]):
            left[:, j] = np.searchsorted(reference[:, j], q[:, j], side="left")
            right[:, j] = m - np.searchsorted(reference[:, j], q[:, j], side="right")

        directional_depth = np.minimum(left, right) / max(float(m), 1.0)
        out[idx] = 1.0 - np.min(directional_depth, axis=1)

    return np.nan_to_num(out, nan=1.0, posinf=1.0, neginf=0.0)

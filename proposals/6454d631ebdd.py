import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions contain tightly constrained residual structure. "
    "OOS inputs can allocate their trailing-direction energy across coordinates unlike the class-specific in-scope spectral profile."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0188"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z_eval = np.asarray(ev.feats, dtype=np.float64)

    n, d = x.shape
    k = tr.logits.shape[1]
    tail_dim = min(64, d)

    counts = np.bincount(y, minlength=k).astype(np.float64)
    means = np.zeros((k, d), dtype=np.float64)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    residuals = x - means[y]
    cov = (residuals.T @ residuals) / max(n - k, 1)
    cov = 0.5 * (cov + cov.T)

    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals[:tail_dim], 1e-8)
    basis = vecs[:, :tail_dim]

    train_tail = (residuals @ basis) / np.sqrt(vals)[None, :]
    train_energy = train_tail * train_tail
    train_profile = train_energy / (train_energy.sum(axis=1, keepdims=True) + 1e-12)

    global_profile = train_profile.mean(axis=0)
    class_profile = np.zeros((k, tail_dim), dtype=np.float64)
    np.add.at(class_profile, y, train_profile)
    class_profile /= np.maximum(counts[:, None], 1.0)

    missing = counts == 0
    class_profile[missing] = global_profile
    class_profile = (class_profile + 1e-5) / (
        class_profile.sum(axis=1, keepdims=True) + 1e-12
    )

    pred = np.argmax(ev.logits, axis=1)
    eval_residuals = z_eval - means[pred]
    eval_tail = (eval_residuals @ basis) / np.sqrt(vals)[None, :]
    eval_energy = eval_tail * eval_tail
    eval_total = eval_energy.sum(axis=1)

    q = eval_energy / (eval_total[:, None] + 1e-12)
    p = class_profile[pred]
    midpoint = 0.5 * (p + q)

    js = 0.5 * np.sum(
        q * np.log((q + 1e-12) / (midpoint + 1e-12)), axis=1
    ) + 0.5 * np.sum(
        p * np.log((p + 1e-12) / (midpoint + 1e-12)), axis=1
    )

    return js + 0.03 * np.log1p(eval_total)

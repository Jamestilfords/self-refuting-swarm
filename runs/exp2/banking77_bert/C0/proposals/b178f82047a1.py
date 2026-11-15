import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS inputs may have unusually large residual energy in low-variance within-class directions. A predicted-class trailing-subspace score, normalized by leading-subspace residual energy, emphasizes this signal while avoiding a ranking identical to an unnormalized trailing residual."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.pooled, dtype=float)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(float)
    sums = np.zeros((k, d), dtype=float)
    np.add.at(sums, y, x)
    global_mean = x.mean(axis=0)
    means = np.divide(
        sums,
        counts[:, None],
        out=np.repeat(global_mean[None, :], k, axis=0),
        where=counts[:, None] > 0,
    )

    centered = x - means[y]
    cov = centered.T @ centered / max(len(x) - k, 1)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)

    tail_dim = max(16, d // 4)
    tail_dim = min(tail_dim, d - 1)
    scale = max(float(np.median(vals)), 1e-8)
    tail_vals = vals[:tail_dim] + 0.02 * scale
    head_vals = vals[tail_dim:] + 0.02 * scale
    tail_vecs = vecs[:, :tail_dim]
    head_vecs = vecs[:, tail_dim:]

    pred = np.argmax(ev.logits, axis=1)
    residual = z - means[pred]
    tail_proj = residual @ tail_vecs
    head_proj = residual @ head_vecs

    tail_energy = np.sum((tail_proj * tail_proj) / tail_vals[None, :], axis=1)
    head_energy = np.sum((head_proj * head_proj) / head_vals[None, :], axis=1)
    scores = np.log1p(tail_energy) - 0.35 * np.log1p(head_energy)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)

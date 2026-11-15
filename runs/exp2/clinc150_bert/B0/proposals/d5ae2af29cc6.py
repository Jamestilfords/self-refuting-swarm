import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Within-class trailing covariance directions encode representation constraints that are weakly "
    "visible in overall distance. Measuring the largest class-conditional standardized violation in "
    "those directions detects localized off-manifold deviations from an assigned intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0133"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(float)
    sums = np.zeros((k, d), dtype=float)
    np.add.at(sums, y, x)
    means = sums / np.maximum(counts[:, None], 1.0)

    centered = x - means[y]
    cov = (centered.T @ centered) / max(x.shape[0] - k, 1)
    cov.flat[::d + 1] += 1e-6 * max(float(np.trace(cov)) / max(d, 1), 1.0)

    vals, vecs = np.linalg.eigh(cov)
    tail_dim = min(64, max(8, d // 8))
    tail_vecs = vecs[:, :tail_dim]

    train_tail = centered @ tail_vecs
    tail_sums = np.zeros((k, tail_dim), dtype=float)
    tail_sq_sums = np.zeros((k, tail_dim), dtype=float)
    np.add.at(tail_sums, y, train_tail)
    np.add.at(tail_sq_sums, y, train_tail * train_tail)

    tail_means = tail_sums / np.maximum(counts[:, None], 1.0)
    tail_vars = tail_sq_sums / np.maximum(counts[:, None], 1.0) - tail_means * tail_means

    global_var = np.maximum(vals[:tail_dim], 1e-8)
    shrink = counts[:, None] / (counts[:, None] + 12.0)
    tail_vars = shrink * tail_vars + (1.0 - shrink) * global_var[None, :]
    tail_vars = np.maximum(tail_vars, 1e-8)

    pred = np.argmax(np.asarray(ev.logits), axis=1)
    eval_tail = (z - means[pred]) @ tail_vecs
    standardized_sq = (eval_tail - tail_means[pred]) ** 2 / tail_vars[pred]

    top = np.max(standardized_sq, axis=1)
    score = np.log1p(np.maximum(top, 0.0))
    return np.nan_to_num(score, nan=0.0, posinf=1e12, neginf=0.0)

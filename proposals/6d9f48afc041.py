import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions are tightly constrained for supported intents. "
    "A predicted-intent-conditioned geometric aggregation of standardized trailing residuals highlights broad unsupported deviations without being dominated by a single coordinate."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0006", "m_0017", "m_0059"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.feats, dtype=np.float64)

    n, d = x.shape
    classes = int(np.max(y)) + 1
    counts = np.bincount(y, minlength=classes).astype(np.float64)

    means = np.zeros((classes, d), dtype=np.float64)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    centered = x - means[y]
    cov = (centered.T @ centered) / max(n - classes, 1)
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Keep a substantial low-variance tail while avoiding numerically null directions.
    tail_dim = max(16, min(64, d // 3))
    tail = eigvecs[:, :tail_dim]

    train_tail = centered @ tail
    global_var = np.mean(train_tail * train_tail, axis=0)
    global_var = np.maximum(global_var, 1e-8)

    class_var = np.zeros((classes, tail_dim), dtype=np.float64)
    np.add.at(class_var, y, train_tail * train_tail)
    class_var /= np.maximum(counts[:, None], 1.0)

    # Shrink sparse intent-specific variances toward the pooled within-class estimate.
    weight = counts[:, None] / (counts[:, None] + 12.0)
    scales = weight * class_var + (1.0 - weight) * global_var[None, :]
    scales = np.maximum(scales, 1e-8)

    train_terms = np.log1p((train_tail * train_tail) / scales[y])
    train_stat = np.mean(train_terms, axis=1)

    stat_mean = np.zeros(classes, dtype=np.float64)
    np.add.at(stat_mean, y, train_stat)
    stat_mean /= np.maximum(counts, 1.0)

    stat_var = np.zeros(classes, dtype=np.float64)
    np.add.at(stat_var, y, (train_stat - stat_mean[y]) ** 2)
    stat_var /= np.maximum(counts, 1.0)
    global_stat_sd = max(float(np.std(train_stat)), 1e-6)
    stat_sd = np.sqrt(np.maximum(stat_var, 0.05 * global_stat_sd * global_stat_sd))

    pred = np.argmax(ev.logits, axis=1).astype(np.int64)
    pred = np.clip(pred, 0, classes - 1)
    eval_tail = (z - means[pred]) @ tail
    eval_stat = np.mean(np.log1p((eval_tail * eval_tail) / scales[pred]), axis=1)

    scores = (eval_stat - stat_mean[pred]) / stat_sd[pred]
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    return scores

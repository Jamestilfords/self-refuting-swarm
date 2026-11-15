import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class eigen-directions represent tightly constrained semantic variation, so a large excursion in even one such direction is evidence against known-intent support. Class-conditional calibration prevents naturally broader intents from being penalized as strongly."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0093", "m_0085"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.feats, dtype=float)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(float)
    counts = np.maximum(counts, 1.0)
    means = np.zeros((k, d), dtype=float)
    np.add.at(means, y, x)
    means /= counts[:, None]

    residuals = x - means[y]
    cov = (residuals.T @ residuals) / max(x.shape[0] - k, 1)
    vals, vecs = np.linalg.eigh(cov)

    q = min(48, d)
    scale = max(float(np.max(vals)), 1.0)
    vals = np.maximum(vals[:q], scale * 1e-7)
    trailing = vecs[:, :q]

    train_proj = (residuals @ trailing) / np.sqrt(vals)
    train_stat = np.max(train_proj * train_proj, axis=1)

    class_sum = np.zeros(k, dtype=float)
    class_sq_sum = np.zeros(k, dtype=float)
    np.add.at(class_sum, y, train_stat)
    np.add.at(class_sq_sum, y, train_stat * train_stat)
    class_mean = class_sum / counts
    class_var = np.maximum(class_sq_sum / counts - class_mean * class_mean, 1e-8)
    class_std = np.sqrt(class_var)

    pred = np.argmax(ev.logits, axis=1)
    eval_residuals = z - means[pred]
    eval_proj = (eval_residuals @ trailing) / np.sqrt(vals)
    eval_stat = np.max(eval_proj * eval_proj, axis=1)

    scores = (eval_stat - class_mean[pred]) / class_std[pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

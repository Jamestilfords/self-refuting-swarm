import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Trailing within-class covariance directions encode tight feature constraints that are weakly represented by total Mahalanobis distance. OOS inputs may produce one or a few unusually large standardized violations in these low-variance directions, so a top-order tail statistic emphasizes localized constraint failures."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0119"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.feats, dtype=float)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(float)
    means = np.zeros((k, d), dtype=float)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    residuals = x - means[y]
    cov = (residuals.T @ residuals) / max(x.shape[0] - k, 1)
    ridge = max(float(np.trace(cov)) / max(d, 1) * 1e-5, 1e-8)
    vals, vecs = np.linalg.eigh(cov + ridge * np.eye(d))

    tail_dim = min(48, d)
    tail_vals = np.maximum(vals[:tail_dim], ridge)
    tail_vecs = vecs[:, :tail_dim]

    pred = np.argmax(ev.logits, axis=1)
    centered = z - means[pred]
    standardized_sq = (centered @ tail_vecs) ** 2 / tail_vals[None, :]

    largest = np.partition(standardized_sq, -3, axis=1)[:, -3:]
    scores = np.mean(largest, axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)

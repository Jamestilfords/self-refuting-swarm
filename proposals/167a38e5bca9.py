import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A class-specific shrunk covariance in a compact pooled-feature subspace captures "
    "intent-dependent variance and correlation structure. OOS inputs are flagged by both "
    "large predicted-class Mahalanobis radius and an atypical disagreement between full and diagonal covariance distances."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    n_train = tr.pooled.shape[0]
    n_classes = tr.logits.shape[1]
    dim = min(64, tr.pooled.shape[1], max(1, n_train - 1))

    projector = ctx.h.pca(tr.pooled, dim)
    z_train = projector.transform(tr.pooled)
    z_eval = projector.transform(ev.pooled)

    global_mean = z_train.mean(axis=0)
    global_centered = z_train - global_mean
    global_var = np.mean(global_centered * global_centered, axis=0)
    global_scale = max(float(np.mean(global_var)), 1e-6)

    means = np.zeros((n_classes, dim), dtype=float)
    chol = np.zeros((n_classes, dim, dim), dtype=float)
    diag_var = np.zeros((n_classes, dim), dtype=float)
    radial_med = np.zeros(n_classes, dtype=float)
    radial_mad = np.ones(n_classes, dtype=float)
    gap_med = np.zeros(n_classes, dtype=float)
    gap_mad = np.ones(n_classes, dtype=float)

    labels = np.asarray(tr.labels, dtype=int)
    eye = np.eye(dim)

    for c in range(n_classes):
        xc = z_train[labels == c]
        if xc.shape[0] == 0:
            xc = z_train

        mu = xc.mean(axis=0)
        centered = xc - mu
        empirical = (centered.T @ centered) / max(xc.shape[0] - 1, 1)

        scale = max(float(np.trace(empirical) / dim), global_scale * 1e-4)
        covariance = 0.80 * empirical + 0.20 * scale * eye
        covariance = covariance + 1e-7 * scale * eye

        means[c] = mu
        chol[c] = np.linalg.cholesky(covariance)
        diag_var[c] = np.maximum(np.diag(covariance), 1e-8)

        solved = np.linalg.solve(chol[c], centered.T)
        full_d = np.sum(solved * solved, axis=0)
        diag_d = np.sum((centered * centered) / diag_var[c], axis=1)
        gap = full_d - diag_d

        radial_med[c] = np.median(full_d)
        radial_mad[c] = max(np.median(np.abs(full_d - radial_med[c])), 1e-4)
        gap_med[c] = np.median(gap)
        gap_mad[c] = max(np.median(np.abs(gap - gap_med[c])), 1e-4)

    predicted = np.argmax(ev.logits, axis=1)
    out = np.empty(z_eval.shape[0], dtype=float)

    for c in range(n_classes):
        idx = np.flatnonzero(predicted == c)
        if idx.size == 0:
            continue

        delta = z_eval[idx] - means[c]
        solved = np.linalg.solve(chol[c], delta.T)
        full_d = np.sum(solved * solved, axis=0)
        diag_d = np.sum((delta * delta) / diag_var[c], axis=1)
        gap = full_d - diag_d

        radial_excess = np.maximum(
            (full_d - radial_med[c]) / radial_mad[c], 0.0
        )
        correlation_anomaly = np.abs(gap - gap_med[c]) / gap_mad[c]
        out[idx] = radial_excess + 0.35 * correlation_anomaly

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=0.0)

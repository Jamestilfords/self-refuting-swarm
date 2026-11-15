import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions have characteristic residual correlations, not merely a characteristic total residual energy. "
    "This score flags examples whose trailing-direction covariance signature is atypical for the classifier-selected intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0188", "m_0220"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    xe = np.asarray(ev.feats, dtype=float)
    k = tr.logits.shape[1]
    d = x.shape[1]
    tail_dim = min(40, max(8, d // 5))

    global_mean = x.mean(axis=0)
    residual = x - global_mean
    classes, counts = np.unique(y, return_counts=True)
    for c, n in zip(classes, counts):
        if n > 0:
            residual[y == c] -= x[y == c].mean(axis=0)

    cov = (residual.T @ residual) / max(len(x) - len(classes), 1)
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    floor = max(float(np.max(vals)) * 1e-8, 1e-10)
    basis = vecs[:, :tail_dim]

    means = np.tile(global_mean @ basis, (k, 1))
    scales = np.ones((k, tail_dim), dtype=float)
    corr = np.zeros((k, tail_dim, tail_dim), dtype=float)

    projected = x @ basis
    global_scale = np.std(projected, axis=0)
    global_scale = np.maximum(global_scale, 1e-6)

    for c in range(k):
        xc = projected[y == c]
        if len(xc) >= 4:
            mu = xc.mean(axis=0)
            sd = np.maximum(xc.std(axis=0), 1e-6)
            zc = np.clip((xc - mu) / sd, -12.0, 12.0)
            rc = (zc.T @ zc) / len(zc)
            rc = 0.85 * rc + 0.15 * np.eye(tail_dim)
            means[c] = mu
            scales[c] = sd
            corr[c] = rc - np.eye(tail_dim)
        else:
            means[c] = global_mean @ basis
            scales[c] = global_scale
            corr[c] = np.zeros((tail_dim, tail_dim), dtype=float)

    pred_train = np.argmax(tr.logits, axis=1)
    q_train = np.empty(len(x), dtype=float)
    for c in range(k):
        mask = pred_train == c
        if np.any(mask):
            z = np.clip((projected[mask] - means[c]) / scales[c], -12.0, 12.0)
            q_train[mask] = np.einsum("ij,jk,ik->i", z, corr[c], z)

    overall_mu = float(np.mean(q_train))
    overall_sd = max(float(np.std(q_train)), 1e-6)
    q_mu = np.full(k, overall_mu, dtype=float)
    q_sd = np.full(k, overall_sd, dtype=float)
    for c in range(k):
        qc = q_train[pred_train == c]
        if len(qc) >= 8:
            q_mu[c] = float(np.mean(qc))
            q_sd[c] = max(float(np.std(qc)), 1e-6)

    pred_eval = np.argmax(ev.logits, axis=1)
    pe = xe @ basis
    out = np.empty(len(xe), dtype=float)
    for c in range(k):
        mask = pred_eval == c
        if np.any(mask):
            z = np.clip((pe[mask] - means[c]) / scales[c], -12.0, 12.0)
            q = np.einsum("ij,jk,ik->i", z, corr[c], z)
            out[mask] = np.abs((q - q_mu[c]) / q_sd[c])

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=0.0)

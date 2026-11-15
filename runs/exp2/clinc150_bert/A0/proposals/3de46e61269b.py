import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A full but shrinkage-regularized covariance for each intent models intent-specific correlation structure rather than assuming one shared Gaussian shape. Queries confidently routed to an intent but violating that intent's correlated pooled-feature geometry receive large conditional residuals."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    dim = min(64, tr.pooled.shape[1])
    pca = ctx.h.pca(tr.pooled, dim)
    xtr = pca.transform(tr.pooled)
    xev = pca.transform(ev.pooled)

    labels = tr.labels.astype(int)
    n_classes = tr.logits.shape[1]
    global_mean = np.mean(xtr, axis=0)
    gc = xtr - global_mean
    global_cov = (gc.T @ gc) / max(len(xtr) - 1, 1)
    base_scale = float(np.trace(global_cov) / dim)
    jitter = max(base_scale, 1e-8) * 1e-5

    predicted = np.argmax(ev.logits, axis=1)
    out = np.empty(len(xev), dtype=float)

    for cls in range(n_classes):
        train_idx = labels == cls
        eval_idx = predicted == cls
        if not np.any(eval_idx):
            continue

        xc = xtr[train_idx]
        n = len(xc)
        if n == 0:
            mu = global_mean
            cov = global_cov
        else:
            mu = np.mean(xc, axis=0)
            centered = xc - mu
            class_cov = (centered.T @ centered) / max(n - 1, 1)
            shrink = dim / (dim + n)
            cov = (1.0 - shrink) * class_cov + shrink * global_cov

        cov = cov + jitter * np.eye(dim)
        inv_cov = np.linalg.inv(cov)
        delta = xev[eval_idx] - mu
        out[eval_idx] = np.einsum("ij,jk,ik->i", delta, inv_cov, delta)

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)

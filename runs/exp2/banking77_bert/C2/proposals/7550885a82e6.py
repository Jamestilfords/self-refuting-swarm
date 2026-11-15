import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions capture subtle deviations from the intent manifolds that leading components can suppress. The score measures distance to the nearest class centroid only in these trailing covariance directions."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    y = np.asarray(tr.labels)
    z = np.asarray(ev.feats, dtype=float)

    classes, inv = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    means = np.zeros((n_classes, x.shape[1]), dtype=float)
    counts = np.bincount(inv, minlength=n_classes).astype(float)
    np.add.at(means, inv, x)
    means /= np.maximum(counts[:, None], 1.0)

    residuals = x - means[inv]
    cov = (residuals.T @ residuals) / max(len(x) - n_classes, 1)
    cov = 0.5 * (cov + cov.T)

    eigvals, eigvecs = np.linalg.eigh(cov)
    n_trailing = min(64, x.shape[1])
    basis = eigvecs[:, :n_trailing]

    train_proj_means = means @ basis
    eval_proj = z @ basis

    eval_norm = np.sum(eval_proj * eval_proj, axis=1, keepdims=True)
    mean_norm = np.sum(train_proj_means * train_proj_means, axis=1)
    dist_sq = eval_norm + mean_norm[None, :] - 2.0 * (eval_proj @ train_proj_means.T)
    scores = np.min(np.maximum(dist_sq, 0.0), axis=1)

    scores = scores + 1e-8 * ctx.h.energy(ev.logits)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)

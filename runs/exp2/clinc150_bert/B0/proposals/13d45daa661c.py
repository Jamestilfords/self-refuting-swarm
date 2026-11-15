import numpy as np
from scipy.special import expit, logsumexp
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "ViM converts the norm outside the principal in-scope feature subspace into a virtual OOS logit, so an input must compete against known-class evidence while also being geometrically supported by the training manifold. This differs from a linear energy-residual sum by using the virtual-class softmax probability."
SOURCE = "https://doi.org/10.48550/arxiv.2203.10807"
PARENTS = []
MEMORY_USED = ["m_0030", "m_0042", "m_0043"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    z = np.asarray(tr.logits, dtype=np.float64)
    xe = np.asarray(ev.feats, dtype=np.float64)
    d = x.shape[1]

    design = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    reg = np.eye(d + 1) * 1e-5
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ z)
    w = coef[:-1].T
    b = coef[-1]
    origin = np.linalg.lstsq(w, -b, rcond=1e-8)[0]

    n_comp = min(128, d, max(1, x.shape[0] - 1))
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=17)
    pca.fit(x)
    basis = pca.components_

    train_shift = x - origin
    train_residual = train_shift - (train_shift @ basis.T) @ basis
    train_norm = np.linalg.norm(train_residual, axis=1)
    denom = max(float(np.mean(train_norm)), 1e-8)
    scale = max(float(np.mean(np.max(z, axis=1))) / denom, 1e-8)

    eval_shift = xe - origin
    eval_residual = eval_shift - (eval_shift @ basis.T) @ basis
    virtual_logit = scale * np.linalg.norm(eval_residual, axis=1)
    known_logsumexp = logsumexp(np.asarray(ev.logits, dtype=np.float64), axis=1)
    scores = expit(virtual_logit - known_logsumexp)
    return np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)

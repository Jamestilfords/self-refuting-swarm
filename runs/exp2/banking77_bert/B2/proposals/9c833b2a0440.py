import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "ViM-style residual energy measures how much a bottleneck representation falls outside the principal in-scope feature subspace. Combining its train-calibrated anomaly with energy catches inputs that are either off-manifold or weakly supported by the classifier."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(128, tr.feats.shape[1], max(1, tr.feats.shape[0] - 1))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=137)
    pca.fit(tr.feats)

    def residual_norm(x):
        centered = x - pca.mean_
        projected = (centered @ pca.components_.T) @ pca.components_
        return np.sqrt(np.sum((centered - projected) ** 2, axis=1))

    train_residual = residual_norm(tr.feats)
    eval_residual = residual_norm(ev.feats)

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    def robust_z(values, train_values):
        med = np.median(train_values)
        scale = np.median(np.abs(train_values - med)) * 1.4826
        scale = max(float(scale), 1e-6)
        return np.clip((values - med) / scale, -20.0, 20.0)

    residual_score = robust_z(eval_residual, train_residual)
    energy_score = robust_z(eval_energy, train_energy)
    return residual_score + 0.35 * energy_score

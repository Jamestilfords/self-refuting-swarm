import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "OOS queries often retain classifier confidence but lie away from the principal feature subspace learned by in-scope examples. Combining this ViM-style residual norm with energy captures both representation-space novelty and logit uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(128, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=137)
    pca.fit(tr.pooled)

    train_centered = tr.pooled - pca.mean_
    eval_centered = ev.pooled - pca.mean_
    train_proj = pca.transform(tr.pooled)
    eval_proj = pca.transform(ev.pooled)

    train_resid_sq = np.maximum(
        np.sum(train_centered * train_centered, axis=1)
        - np.sum(train_proj * train_proj, axis=1),
        0.0,
    )
    eval_resid_sq = np.maximum(
        np.sum(eval_centered * eval_centered, axis=1)
        - np.sum(eval_proj * eval_proj, axis=1),
        0.0,
    )

    train_resid = np.sqrt(train_resid_sq)
    eval_resid = np.sqrt(eval_resid_sq)
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    resid_scale = np.std(train_resid) + 1e-8
    energy_scale = np.std(train_energy) + 1e-8

    scores = (
        (eval_resid - np.mean(train_resid)) / resid_scale
        + (eval_energy - np.mean(train_energy)) / energy_scale
    )
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

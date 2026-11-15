import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "A ViM-style residual measures encoder content outside the principal in-scope feature subspace, "
    "while train-calibrated energy retains classifier uncertainty information for residual-supported inputs."
)
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0004", "m_0147"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    n, d = x_train.shape
    rank = max(1, min(128, n, d))
    pca = PCA(n_components=rank, svd_solver="randomized", random_state=2718)
    pca.fit(x_train)

    train_centered = x_train - pca.mean_
    eval_centered = x_eval - pca.mean_
    train_proj = pca.transform(x_train)
    eval_proj = pca.transform(x_eval)

    train_res2 = np.maximum(
        np.sum(train_centered * train_centered, axis=1)
        - np.sum(train_proj * train_proj, axis=1),
        0.0,
    )
    eval_res2 = np.maximum(
        np.sum(eval_centered * eval_centered, axis=1)
        - np.sum(eval_proj * eval_proj, axis=1),
        0.0,
    )
    train_res = np.sqrt(train_res2)
    eval_res = np.sqrt(eval_res2)

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=np.float64)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=np.float64)

    res_med = np.median(train_res)
    res_scale = np.median(np.abs(train_res - res_med)) * 1.4826 + 1e-8
    en_med = np.median(train_energy)
    en_scale = np.median(np.abs(train_energy - en_med)) * 1.4826 + 1e-8

    residual_surprise = np.clip((eval_res - res_med) / res_scale, -20.0, 20.0)
    energy_surprise = np.clip((eval_energy - en_med) / en_scale, -20.0, 20.0)
    return energy_surprise + residual_surprise

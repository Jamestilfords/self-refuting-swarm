import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Queries with large residual norm outside the principal training-feature subspace are likely unlike in-scope representations. Calibrating this residual against training energy combines representation atypicality with classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    pca = PCA(
        n_components=256,
        svd_solver="randomized",
        random_state=173,
        iterated_power=3,
    )
    z_train = pca.fit_transform(tr.pooled)
    z_eval = pca.transform(ev.pooled)

    train_centered = tr.pooled - pca.mean_
    eval_centered = ev.pooled - pca.mean_

    train_res2 = np.maximum(
        np.sum(train_centered * train_centered, axis=1)
        - np.sum(z_train * z_train, axis=1),
        0.0,
    )
    eval_res2 = np.maximum(
        np.sum(eval_centered * eval_centered, axis=1)
        - np.sum(z_eval * z_eval, axis=1),
        0.0,
    )

    train_res = np.sqrt(train_res2)
    eval_res = np.sqrt(eval_res2)

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    res_scale = np.std(train_res) + 1e-8
    energy_scale = np.std(train_energy) + 1e-8

    residual_z = (eval_res - np.mean(train_res)) / res_scale
    energy_z = (eval_energy - np.mean(train_energy)) / energy_scale

    return 0.65 * residual_z + 0.35 * energy_z

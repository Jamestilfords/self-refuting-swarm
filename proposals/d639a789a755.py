import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy captures uncertain classifier responses, while PCA residual magnitude measures deviation from the shared in-scope encoder subspace. Train-fitted empirical tail maps put these complementary signals on a common bounded scale without using evaluation statistics."
SOURCE = "https://deel-ai.github.io/oodeel/latest/notebooks/torch/demo_vim_torch/"
PARENTS = ["energy"]
MEMORY_USED = ["m_0165", "m_0160"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.nan_to_num(tr.pooled, nan=0.0, posinf=1e6, neginf=-1e6)
    x_eval = np.nan_to_num(ev.pooled, nan=0.0, posinf=1e6, neginf=-1e6)

    dim = min(64, x_train.shape[1], max(1, x_train.shape[0] - 1))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=173)
    z_train = pca.fit_transform(x_train)
    z_eval = pca.transform(x_eval)

    train_centered = x_train - pca.mean_
    eval_centered = x_eval - pca.mean_
    train_residual = np.sqrt(np.maximum(
        np.sum(train_centered * train_centered, axis=1) -
        np.sum(z_train * z_train, axis=1),
        0.0,
    ))
    eval_residual = np.sqrt(np.maximum(
        np.sum(eval_centered * eval_centered, axis=1) -
        np.sum(z_eval * z_eval, axis=1),
        0.0,
    ))

    train_energy = np.nan_to_num(
        ctx.h.energy(tr.logits), nan=0.0, posinf=1e6, neginf=-1e6
    )
    eval_energy = np.nan_to_num(
        ctx.h.energy(ev.logits), nan=0.0, posinf=1e6, neginf=-1e6
    )

    energy_ref = np.sort(train_energy)
    residual_ref = np.sort(train_residual)
    denom = float(len(train_energy) + 1)

    energy_tail = np.searchsorted(energy_ref, eval_energy, side="right") / denom
    residual_tail = np.searchsorted(residual_ref, eval_residual, side="right") / denom

    return 0.5 * energy_tail + 0.5 * residual_tail

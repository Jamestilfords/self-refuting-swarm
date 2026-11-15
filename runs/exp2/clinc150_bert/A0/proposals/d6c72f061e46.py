import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "The norm outside a train-derived principal representation subspace measures unsupported feature content, while energy retains classifier uncertainty information. Train-only robust scaling lets residual novelty contribute without being dominated by feature magnitude."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0027"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    n_comp = min(128, x_train.shape[0] - 1, x_train.shape[1])

    pca = PCA(n_components=max(1, n_comp), svd_solver="randomized", random_state=2718)
    pca.fit(x_train)

    def residual_norm(x):
        centered = x - pca.mean_
        projected = pca.transform(x)
        total_sq = np.sum(centered * centered, axis=1)
        explained_sq = np.sum(projected * projected, axis=1)
        return np.sqrt(np.maximum(total_sq - explained_sq, 0.0))

    train_residual = residual_norm(x_train)
    eval_residual = residual_norm(x_eval)

    res_med = np.median(train_residual)
    res_scale = np.median(np.abs(train_residual - res_med)) * 1.4826 + 1e-8

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    en_med = np.median(train_energy)
    en_scale = np.median(np.abs(train_energy - en_med)) * 1.4826 + 1e-8

    residual_z = (eval_residual - res_med) / res_scale
    energy_z = (eval_energy - en_med) / en_scale
    return energy_z + 0.5 * residual_z

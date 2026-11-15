import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "A pooled representation that has large energy outside the train-fitted principal subspace is poorly explained by in-scope feature structure. Averaging its train-calibrated residual percentile with energy retains classifier uncertainty while emphasizing unexplained feature content."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0025"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    n_comp = min(256, x_train.shape[0], x_train.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=137)
    pca.fit(x_train)

    train_centered = x_train - pca.mean_
    eval_centered = x_eval - pca.mean_
    train_proj = pca.transform(x_train)
    eval_proj = pca.transform(x_eval)

    train_residual = np.maximum(
        np.einsum("ij,ij->i", train_centered, train_centered)
        - np.einsum("ij,ij->i", train_proj, train_proj),
        0.0,
    )
    eval_residual = np.maximum(
        np.einsum("ij,ij->i", eval_centered, eval_centered)
        - np.einsum("ij,ij->i", eval_proj, eval_proj),
        0.0,
    )

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=np.float64)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=np.float64)

    residual_ref = np.sort(train_residual)
    energy_ref = np.sort(train_energy)
    residual_pct = np.searchsorted(residual_ref, eval_residual, side="right") / float(len(residual_ref))
    energy_pct = np.searchsorted(energy_ref, eval_energy, side="right") / float(len(energy_ref))

    return 0.5 * residual_pct + 0.5 * energy_pct

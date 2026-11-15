import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "A principal in-scope subspace explains common encoder variation, but residual magnitude should also be judged relative to how far the query lies within that subspace. Training-calibrated conditional residual excess highlights unsupported feature components without using evaluation-set statistics."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0004", "m_0015"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)

    n_train, dim = x_train.shape
    n_comp = max(1, min(128, dim, n_train))
    pca = PCA(
        n_components=n_comp,
        svd_solver="randomized",
        random_state=271828,
        iterated_power=4,
    )
    z_train = pca.fit_transform(x_train)
    z_eval = pca.transform(x_eval)

    train_centered = x_train - pca.mean_
    eval_centered = x_eval - pca.mean_

    train_total_sq = np.einsum("ij,ij->i", train_centered, train_centered)
    eval_total_sq = np.einsum("ij,ij->i", eval_centered, eval_centered)
    train_proj_sq = np.einsum("ij,ij->i", z_train, z_train)
    eval_proj_sq = np.einsum("ij,ij->i", z_eval, z_eval)

    train_resid_sq = np.maximum(train_total_sq - train_proj_sq, 0.0)
    eval_resid_sq = np.maximum(eval_total_sq - eval_proj_sq, 0.0)

    q_train = np.log1p(np.maximum(train_proj_sq, 0.0))
    q_eval = np.log1p(np.maximum(eval_proj_sq, 0.0))
    y_train = np.log1p(train_resid_sq)

    q_mean = float(np.mean(q_train))
    q_scale = float(np.std(q_train))
    if q_scale < 1e-8:
        q_scale = 1.0

    u_train = (q_train - q_mean) / q_scale
    u_eval = (q_eval - q_mean) / q_scale
    design_train = np.column_stack(
        (np.ones(n_train), u_train, u_train * u_train)
    )
    design_eval = np.column_stack(
        (np.ones(len(x_eval)), u_eval, u_eval * u_eval)
    )

    gram = design_train.T @ design_train
    penalty = np.diag([0.0, 1e-3, 1e-3])
    beta = np.linalg.solve(gram + penalty, design_train.T @ y_train)

    train_excess = y_train - design_train @ beta
    eval_excess = np.log1p(eval_resid_sq) - design_eval @ beta

    center = float(np.median(train_excess))
    scale = float(np.median(np.abs(train_excess - center))) * 1.4826
    if scale < 1e-6:
        scale = float(np.std(train_excess))
    if scale < 1e-6:
        scale = 1.0

    scores = (eval_excess - center) / scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

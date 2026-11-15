import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "The ratio of off-subspace residual energy to principal-subspace energy detects representations that point away from the in-scope feature manifold, rather than merely having large feature magnitude. This angular residual-space novelty differs from standard ViM's additive residual-logit score."
SOURCE = "https://openaccess.thecvf.com/content/CVPR2022/html/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.html"
PARENTS = []
MEMORY_USED = ["m_0067", "m_0068"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(ev.pooled, dtype=np.float64)

    mean = np.mean(x, axis=0)
    xc = x - mean
    yc = y - mean

    cov = (xc.T @ xc) / max(x.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]

    rank = min(128, x.shape[1] - 1)
    basis = vecs[:, order[:rank]]

    train_proj_sq = np.sum((xc @ basis) ** 2, axis=1)
    train_total_sq = np.sum(xc * xc, axis=1)
    train_res_sq = np.maximum(train_total_sq - train_proj_sq, 0.0)
    floor = np.median(train_res_sq) + 1e-12

    eval_proj_sq = np.sum((yc @ basis) ** 2, axis=1)
    eval_total_sq = np.sum(yc * yc, axis=1)
    eval_res_sq = np.maximum(eval_total_sq - eval_proj_sq, 0.0)

    return np.log1p(eval_res_sq / (eval_proj_sq + floor))

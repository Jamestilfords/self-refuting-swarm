import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A virtual OOD logit converts unsupported pooled-feature residuals into competition against the classifier logits. The resulting OOD probability retains the logit-energy ordering when residual support is strong, while large residuals dominate for energy-uncertain queries."
SOURCE = "https://openaccess.thecvf.com/content/CVPR2022/html/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.html"
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0034", "m_0022"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    origin = np.mean(tr.pooled, axis=0, keepdims=True)
    centered_train = tr.pooled - origin
    centered_eval = ev.pooled - origin

    dim = min(128, centered_train.shape[1], max(1, centered_train.shape[0] - 1))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=137)
    pca.fit(centered_train)

    train_proj = pca.inverse_transform(pca.transform(centered_train))
    eval_proj = pca.inverse_transform(pca.transform(centered_eval))
    train_residual = np.linalg.norm(centered_train - train_proj, axis=1)
    eval_residual = np.linalg.norm(centered_eval - eval_proj, axis=1)

    denom = float(np.mean(train_residual))
    denom = max(denom, 1e-12)
    logit_scale = float(np.mean(np.abs(np.max(tr.logits, axis=1))))
    alpha = max(logit_scale / denom, 1e-6)

    virtual_logit = alpha * eval_residual
    logits = ev.logits
    row_max = np.maximum(np.max(logits, axis=1), virtual_logit)
    log_denom = row_max + np.log(
        np.exp(logits - row_max[:, None]).sum(axis=1)
        + np.exp(virtual_logit - row_max)
    )
    scores = virtual_logit - log_denom
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=-50.0)

import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "ViM-style residual norms measure feature content outside the principal in-scope representation subspace. Comparing a train-calibrated virtual residual logit against classifier log-partition confidence highlights samples that are both off-manifold and weakly supported by known intents."
SOURCE = "https://openaccess.thecvf.com/content/CVPR2022/html/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.html"
PARENTS = []
MEMORY_USED = ["m_0046"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    mean = np.mean(tr.pooled, axis=0)
    max_components = min(256, tr.pooled.shape[1], tr.pooled.shape[0] - 1)
    pca = PCA(
        n_components=max_components,
        svd_solver="randomized",
        random_state=314159,
    )
    pca.fit(tr.pooled)

    tr_proj = pca.transform(tr.pooled)
    ev_proj = pca.transform(ev.pooled)

    tr_centered_sq = np.sum((tr.pooled - mean) ** 2, axis=1)
    ev_centered_sq = np.sum((ev.pooled - mean) ** 2, axis=1)
    tr_residual = np.sqrt(np.maximum(tr_centered_sq - np.sum(tr_proj ** 2, axis=1), 0.0))
    ev_residual = np.sqrt(np.maximum(ev_centered_sq - np.sum(ev_proj ** 2, axis=1), 0.0))

    residual_scale = max(float(np.mean(tr_residual)), 1e-8)
    virtual_scale = float(np.mean(np.max(tr.logits, axis=1))) / residual_scale
    if not np.isfinite(virtual_scale):
        virtual_scale = 1.0

    log_partition = np.logaddexp.reduce(ev.logits, axis=1)
    scores = virtual_scale * ev_residual - log_partition
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)

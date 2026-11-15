import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "ViM-style residual norm measures deviation from the principal in-scope encoder subspace, while a virtual logit compares that deviation directly against the classifier's known-intent evidence. OOS inputs should receive a large virtual-class probability when their representation has substantial non-principal residual energy."
SOURCE = "https://openaccess.thecvf.com/content/CVPR2022/html/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.html"
PARENTS = []
MEMORY_USED = ["m_0045"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    dim = min(128, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    mean = np.mean(tr.pooled, axis=0)
    pca = ctx.h.pca(tr.pooled, dim)

    tr_centered = tr.pooled - mean
    ev_centered = ev.pooled - mean
    tr_proj = pca.transform(tr.pooled)
    ev_proj = pca.transform(ev.pooled)

    tr_res_sq = np.maximum(
        np.sum(tr_centered * tr_centered, axis=1) - np.sum(tr_proj * tr_proj, axis=1),
        0.0,
    )
    ev_res_sq = np.maximum(
        np.sum(ev_centered * ev_centered, axis=1) - np.sum(ev_proj * ev_proj, axis=1),
        0.0,
    )
    tr_res = np.sqrt(tr_res_sq)
    ev_res = np.sqrt(ev_res_sq)

    alpha = np.mean(np.max(tr.logits, axis=1)) / max(np.mean(tr_res), 1e-12)
    virtual = alpha * ev_res

    max_logit = np.maximum(np.max(ev.logits, axis=1), virtual)
    log_known = max_logit + np.log(np.sum(np.exp(ev.logits - max_logit[:, None]), axis=1))
    return 1.0 / (1.0 + np.exp(np.clip(log_known - virtual, -80.0, 80.0)))

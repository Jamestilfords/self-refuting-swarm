import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "A ViM-style residual norm measures feature energy outside the dominant in-scope principal subspace. "
    "Conditioning its robust calibration on the classifier-predicted intent accounts for classes with different residual variability."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0093"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    k = min(128, x_train.shape[1], max(1, x_train.shape[0] - 1))

    pca = PCA(
        n_components=k,
        svd_solver="randomized",
        random_state=2718,
    )
    pca.fit(x_train)

    center = pca.mean_
    comp = pca.components_

    train_centered = x_train - center
    eval_centered = x_eval - center

    train_proj = train_centered @ comp.T
    eval_proj = eval_centered @ comp.T

    train_resid_sq = np.maximum(
        np.sum(train_centered * train_centered, axis=1)
        - np.sum(train_proj * train_proj, axis=1),
        0.0,
    )
    eval_resid_sq = np.maximum(
        np.sum(eval_centered * eval_centered, axis=1)
        - np.sum(eval_proj * eval_proj, axis=1),
        0.0,
    )

    train_resid = np.sqrt(train_resid_sq)
    eval_resid = np.sqrt(eval_resid_sq)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_med = np.median(train_resid)
    global_mad = np.median(np.abs(train_resid - global_med))
    global_scale = max(1.4826 * global_mad, 1e-6)

    medians = np.full(n_classes, global_med, dtype=np.float64)
    scales = np.full(n_classes, global_scale, dtype=np.float64)

    for cls in range(n_classes):
        values = train_resid[train_pred == cls]
        if values.size >= 8:
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            local_scale = max(1.4826 * mad, 1e-6)
            weight = values.size / (values.size + 24.0)
            medians[cls] = weight * med + (1.0 - weight) * global_med
            scales[cls] = weight * local_scale + (1.0 - weight) * global_scale

    scores = (eval_resid - medians[eval_pred]) / scales[eval_pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

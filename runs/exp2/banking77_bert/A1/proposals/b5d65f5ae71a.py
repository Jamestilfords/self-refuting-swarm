import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A multi-scale cosine neighbourhood score combines the radius needed to reach many in-scope examples with local neighbourhood expansion. OOS queries should be both farther from in-scope support and less locally concentrated than typical training queries."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    dim = min(64, tr.feats.shape[1])
    pca = ctx.h.pca(tr.feats, dim)
    x_train = ctx.h.l2(pca.transform(tr.feats))
    x_eval = ctx.h.l2(pca.transform(ev.feats))

    n_train = x_train.shape[0]
    k_outer = min(48, max(2, n_train - 1))
    k_inner = min(8, k_outer - 1)

    nn = NearestNeighbors(
        n_neighbors=k_outer + 1,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    ).fit(x_train)

    train_dist = nn.kneighbors(x_train, return_distance=True)[0][:, 1:]
    eval_dist = nn.kneighbors(x_eval, n_neighbors=k_outer, return_distance=True)[0]

    eps = 1e-8
    train_raw = (
        train_dist[:, k_outer - 1]
        + 0.35
        * np.log((train_dist[:, k_outer - 1] + eps) /
                 (train_dist[:, k_inner - 1] + eps))
    )
    eval_raw = (
        eval_dist[:, k_outer - 1]
        + 0.35
        * np.log((eval_dist[:, k_outer - 1] + eps) /
                 (eval_dist[:, k_inner - 1] + eps))
    )

    center = np.median(train_raw)
    scale = np.median(np.abs(train_raw - center)) * 1.4826 + eps
    return np.nan_to_num((eval_raw - center) / scale, nan=0.0, posinf=1e6, neginf=-1e6)

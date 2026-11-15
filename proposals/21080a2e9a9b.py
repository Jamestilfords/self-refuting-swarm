import numpy as np
from scipy.spatial import cKDTree

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A local-outlier-factor score compares a query's neighborhood radius with the intrinsic neighborhood radii of its nearest in-scope examples. This flags representations that may have a moderate nearest-neighbor distance but lie in a region substantially sparser than any learned intent manifold."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    pca = ctx.h.pca(tr.feats, 24)
    x_train = pca.transform(tr.feats)
    x_eval = pca.transform(ev.feats)

    scale = np.std(x_train, axis=0)
    scale = np.maximum(scale, 1e-6)
    x_train = x_train / scale
    x_eval = x_eval / scale

    k = min(24, max(2, len(x_train) - 1))
    tree = cKDTree(x_train)

    train_dist, train_ind = tree.query(x_train, k=k + 1)
    neighbor_dist = train_dist[:, 1:]
    neighbor_ind = train_ind[:, 1:]

    kth_radius = neighbor_dist[:, -1]
    train_reach = np.maximum(neighbor_dist, kth_radius[neighbor_ind])
    train_lrd = 1.0 / np.maximum(np.mean(train_reach, axis=1), 1e-8)

    eval_dist, eval_ind = tree.query(x_eval, k=k)
    if k == 1:
        eval_dist = eval_dist[:, None]
        eval_ind = eval_ind[:, None]

    eval_reach = np.maximum(eval_dist, kth_radius[eval_ind])
    eval_lrd = 1.0 / np.maximum(np.mean(eval_reach, axis=1), 1e-8)
    lof = np.mean(train_lrd[eval_ind], axis=1) / np.maximum(eval_lrd, 1e-8)

    return np.log(np.maximum(lof, 1e-8))

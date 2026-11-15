import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "A local neighborhood with both large average distance and mixed intent labels indicates weak "
    "in-scope support: OOS queries should be farther from training examples and less class-pure locally."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0069"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(48, tr.feats.shape[1], tr.feats.shape[0])
    pca = ctx.h.pca(tr.feats, dim)
    train_x = pca.transform(tr.feats)
    eval_x = pca.transform(ev.feats)

    center = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    scale = np.maximum(scale, 1e-6)

    train_x = np.nan_to_num((train_x - center) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    eval_x = np.nan_to_num((eval_x - center) / scale, nan=0.0, posinf=0.0, neginf=0.0)

    k = min(48, train_x.shape[0])
    nn = NearestNeighbors(
        n_neighbors=k,
        algorithm="brute",
        metric="euclidean",
        n_jobs=1,
    )
    nn.fit(train_x)
    distances, indices = nn.kneighbors(eval_x, return_distance=True)

    classes = np.unique(tr.labels)
    neighbor_classes = np.searchsorted(classes, tr.labels[indices])
    counts = np.zeros((eval_x.shape[0], classes.shape[0]), dtype=float)
    np.add.at(
        counts,
        (np.arange(eval_x.shape[0])[:, None], neighbor_classes),
        1.0,
    )

    probs = counts / float(k)
    entropy = -np.sum(np.where(probs > 0.0, probs * np.log(probs), 0.0), axis=1)
    if k > 1:
        entropy /= np.log(min(k, classes.shape[0]))
    else:
        entropy.fill(0.0)

    mean_distance = np.mean(distances, axis=1)
    return np.log1p(mean_distance) * (1.0 + 0.5 * entropy)

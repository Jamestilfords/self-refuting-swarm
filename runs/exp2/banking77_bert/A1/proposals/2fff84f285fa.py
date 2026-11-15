import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = "OOS queries should be both distant from the in-scope cosine neighbourhood and poorly supported by neighbours sharing the classifier's predicted intent. Local predicted-intent agreement adds neighbourhood-label structure that a distance-only kNN score misses."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = 16

    train_x = ctx.h.l2(tr.pooled)
    eval_x = ctx.h.l2(ev.pooled)

    train_radius = ctx.h.knn(tr.pooled, tr.pooled, k=k + 1, l2=True)
    center = float(np.median(train_radius))
    scale = float(np.median(np.abs(train_radius - center)))
    scale = max(scale, 1e-6)

    nn = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
        algorithm="brute",
    )
    nn.fit(train_x)
    distances, indices = nn.kneighbors(eval_x, return_distance=True)

    radius = distances[:, -1]
    distance_score = (radius - center) / scale

    predicted = np.argmax(ev.logits, axis=1)
    neighbor_labels = tr.labels[indices]
    agreement = np.mean(neighbor_labels == predicted[:, None], axis=1)

    counts = np.zeros((len(ev.pooled),), dtype=float)
    for j in range(k):
        counts += (neighbor_labels[:, j] == predicted).astype(float)
    agreement = counts / float(k)

    score = distance_score + 1.5 * (1.0 - agreement)
    return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

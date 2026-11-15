import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "A query is suspicious when its nearest encoder neighbors do not support the intent predicted "
    "by the classifier. Local label disagreement captures representation-logit inconsistency rather "
    "than relying only on distance to the predicted class."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0060", "m_0061"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    z_tr = ctx.h.l2(tr.pooled)
    z_ev = ctx.h.l2(ev.pooled)

    k = min(40, len(z_tr))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(z_tr)
    distances, indices = nn.kneighbors(z_ev)

    predicted = np.argmax(ev.logits, axis=1)
    neighbor_labels = tr.labels[indices]
    support = np.mean(neighbor_labels == predicted[:, None], axis=1)

    mean_distance = np.mean(distances, axis=1)
    return (1.0 - support) + 0.02 * mean_distance

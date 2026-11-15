import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "edl"
RATIONALE = (
    "The normalized Dirichlet concentration profile describes how evidence is distributed across intents, independently of total evidence or vacuity. "
    "A query whose predicted-intent profile is angularly unlike profiles seen for that intent is likely out-of-scope."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0049"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    a_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), eps)

    p_tr = a_tr / np.sum(a_tr, axis=1, keepdims=True)
    p_ev = a_ev / np.sum(a_ev, axis=1, keepdims=True)

    clr_tr = np.log(p_tr)
    clr_ev = np.log(p_ev)
    clr_tr = clr_tr - np.mean(clr_tr, axis=1, keepdims=True)
    clr_ev = clr_ev - np.mean(clr_ev, axis=1, keepdims=True)

    z_tr = clr_tr / np.maximum(np.linalg.norm(clr_tr, axis=1, keepdims=True), eps)
    z_ev = clr_ev / np.maximum(np.linalg.norm(clr_ev, axis=1, keepdims=True), eps)

    pred_ev = np.argmax(ev.logits, axis=1)
    scores = np.empty(z_ev.shape[0], dtype=float)

    global_nn = NearestNeighbors(
        n_neighbors=min(5, z_tr.shape[0]),
        metric="euclidean",
        algorithm="auto",
    ).fit(z_tr)

    for c in np.unique(pred_ev):
        mask = pred_ev == c
        class_x = z_tr[tr.labels == c]
        if class_x.shape[0] >= 2:
            k = min(5, class_x.shape[0])
            nn = NearestNeighbors(
                n_neighbors=k,
                metric="euclidean",
                algorithm="auto",
            ).fit(class_x)
        else:
            k = min(5, z_tr.shape[0])
            nn = global_nn
        scores[mask] = nn.kneighbors(
            z_ev[mask], n_neighbors=k, return_distance=True
        )[0][:, -1]

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)

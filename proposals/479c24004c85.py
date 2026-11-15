import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "edl"
RATIONALE = (
    "The relative Dirichlet concentration pattern encodes evidential competition independently "
    "of total evidence. In-scope queries should have nearby class-conditional concentration shapes, "
    "while OOS queries can induce unfamiliar allocations even at similar vacuity."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0116", "m_0050", "m_0064"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    alpha_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    alpha_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    # Centered log-ratio coordinates remove the overall evidence scale and
    # retain only the composition of evidence across intents.
    z_tr = np.log(alpha_tr)
    z_tr = z_tr - z_tr.mean(axis=1, keepdims=True)
    z_ev = np.log(alpha_ev)
    z_ev = z_ev - z_ev.mean(axis=1, keepdims=True)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    n_eval = z_ev.shape[0]
    out = np.empty(n_eval, dtype=float)

    # A global fallback remains train-fitted for rarely predicted intents.
    global_k = min(10, len(z_tr))
    global_nn = NearestNeighbors(n_neighbors=global_k, metric="euclidean")
    global_nn.fit(z_tr)
    global_dist = global_nn.kneighbors(z_ev, return_distance=True)[0][:, -1]

    for c in np.unique(pred_ev):
        test_idx = np.flatnonzero(pred_ev == c)
        train_idx = np.flatnonzero(pred_tr == c)

        if len(train_idx) >= 3:
            k = min(10, len(train_idx))
            nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
            nn.fit(z_tr[train_idx])
            d = nn.kneighbors(z_ev[test_idx], return_distance=True)[0][:, -1]

            # Normalize by a train-only class scale so sparse and dense
            # predicted-intent manifolds are comparable.
            ref = nn.kneighbors(z_tr[train_idx], return_distance=True)[0][:, -1]
            scale = np.median(ref) + 1e-6
            out[test_idx] = d / scale
        else:
            out[test_idx] = global_dist[test_idx]

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)

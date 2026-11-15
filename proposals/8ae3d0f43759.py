import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Predictive entropy captures uncertain intent competition, while distance to the classifier-selected "
    "intent centroid captures representation support. Class-conditional training empirical tails put both "
    "signals on a shared scale without using evaluation statistics."
)
SOURCE = "https://hanleizhang.github.io/files/TASLP23-DA-ADB/DA-ADB.pdf"
PARENTS = []
MEMORY_USED = ["m_0133"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]

    def entropy(logits):
        p = ctx.h.softmax(logits)
        return -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    centroids = np.zeros((k, tr.pooled.shape[1]), dtype=float)
    global_center = np.mean(tr.pooled, axis=0)
    for c in range(k):
        members = tr.pooled[tr.labels == c]
        centroids[c] = np.mean(members, axis=0) if len(members) else global_center

    train_dist = np.linalg.norm(tr.pooled - centroids[train_pred], axis=1)
    eval_dist = np.linalg.norm(ev.pooled - centroids[eval_pred], axis=1)
    train_ent = entropy(tr.logits)
    eval_ent = entropy(ev.logits)

    global_dist_sorted = np.sort(train_dist)
    global_ent_sorted = np.sort(train_ent)
    out = np.empty(len(ev.logits), dtype=float)

    for c in range(k):
        mask_tr = train_pred == c
        mask_ev = eval_pred == c
        if not np.any(mask_ev):
            continue

        dist_ref = np.sort(train_dist[mask_tr]) if np.sum(mask_tr) >= 8 else global_dist_sorted
        ent_ref = np.sort(train_ent[mask_tr]) if np.sum(mask_tr) >= 8 else global_ent_sorted

        dist_tail = np.searchsorted(dist_ref, eval_dist[mask_ev], side="right") / len(dist_ref)
        ent_tail = np.searchsorted(ent_ref, eval_ent[mask_ev], side="right") / len(ent_ref)
        out[mask_ev] = 0.5 * (dist_tail + ent_tail)

    return out

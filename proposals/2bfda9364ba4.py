import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Predictive entropy captures unusual classifier competition, while the Mahalanobis "
    "distance to the classifier-selected intent captures whether that decision has pooled-feature support."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0021", "m_0128"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_all_dist = g.class_mahalanobis_sq(tr.pooled)
    eval_all_dist = g.class_mahalanobis_sq(ev.pooled)

    train_dist = train_all_dist[np.arange(len(tr.labels)), train_pred]
    eval_dist = eval_all_dist[np.arange(len(eval_pred)), eval_pred]

    train_entropy = ctx.h.entropy(tr.logits)
    eval_entropy = ctx.h.entropy(ev.logits)

    global_entropy = np.sort(train_entropy)
    global_dist = np.sort(train_dist)
    entropy_tail = np.empty(len(eval_pred), dtype=float)
    dist_tail = np.empty(len(eval_pred), dtype=float)

    n_classes = tr.logits.shape[1]
    for c in range(n_classes):
        train_mask = train_pred == c
        eval_mask = eval_pred == c

        if not np.any(eval_mask):
            continue

        if np.sum(train_mask) >= 8:
            entropy_ref = np.sort(train_entropy[train_mask])
            dist_ref = np.sort(train_dist[train_mask])
        else:
            entropy_ref = global_entropy
            dist_ref = global_dist

        entropy_tail[eval_mask] = (
            np.searchsorted(entropy_ref, eval_entropy[eval_mask], side="right")
            / float(len(entropy_ref))
        )
        dist_tail[eval_mask] = (
            np.searchsorted(dist_ref, eval_dist[eval_mask], side="right")
            / float(len(dist_ref))
        )

    entropy_tail = np.clip(entropy_tail, 0.0, 1.0 - 1e-12)
    dist_tail = np.clip(dist_tail, 0.0, 1.0 - 1e-12)

    return -np.log1p(-entropy_tail) - np.log1p(-dist_tail)

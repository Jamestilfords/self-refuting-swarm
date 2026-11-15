import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy is preserved unchanged for train-calibrated confident queries, while increasingly "
    "uncertain queries receive a predicted-intent prototype-support penalty. This lets feature "
    "geometry distinguish unsupported representations specifically where logits are ambiguous."
)
SOURCE = ""
PARENTS = ["energy", "d_port"]
MEMORY_USED = ["m_0115"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    global_center = np.mean(x_train, axis=0)

    centers = np.empty((n_classes, x_train.shape[1]), dtype=float)
    for c in range(n_classes):
        members = x_train[tr.labels == c]
        centers[c] = np.mean(members, axis=0) if len(members) else global_center

    train_dist = np.sum((x_train - centers[train_pred]) ** 2, axis=1)
    eval_dist = np.sum((x_eval - centers[eval_pred]) ** 2, axis=1)

    support_tail = np.empty(len(x_eval), dtype=float)
    for c in range(n_classes):
        ref = np.sort(train_dist[train_pred == c])
        mask = eval_pred == c
        if len(ref) < 8:
            ref = np.sort(train_dist)
        if np.any(mask):
            support_tail[mask] = np.searchsorted(ref, eval_dist[mask], side="right") / (
                len(ref) + 1.0
            )

    q_confident = float(np.quantile(train_energy, 0.60))
    q_uncertain = float(np.quantile(train_energy, 0.90))
    scale = max(q_uncertain - q_confident, 1e-8)
    gate = np.clip((eval_energy - q_confident) / scale, 0.0, 1.0)

    return eval_energy + 2.0 * gate * support_tail

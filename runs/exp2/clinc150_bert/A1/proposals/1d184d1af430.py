import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "L2-normalized class prototypes summarize the semantic direction of each labeled in-scope intent. "
    "A query with unusually low cosine alignment to its routed intent prototype, relative to that intent's training distribution, is more likely OOS."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0123"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = tr.labels.astype(int)
    routed_eval = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes)
    for c in range(n_classes):
        if counts[c] > 0:
            prototypes[c] = x_train[labels == c].mean(axis=0)
    prototypes = ctx.h.l2(prototypes)

    train_sim = np.sum(x_train * prototypes[labels], axis=1)
    global_mean = float(np.mean(train_sim))
    global_scale = float(np.std(train_sim)) + 1e-6

    class_mean = np.full(n_classes, global_mean, dtype=float)
    class_scale = np.full(n_classes, global_scale, dtype=float)
    for c in range(n_classes):
        vals = train_sim[labels == c]
        if vals.size >= 2:
            class_mean[c] = float(np.mean(vals))
            class_scale[c] = float(np.std(vals)) + 1e-6

    eval_sim = np.sum(x_eval * prototypes[routed_eval], axis=1)
    scores = (class_mean[routed_eval] - eval_sim) / class_scale[routed_eval]

    margin = np.partition(ev.logits, -2, axis=1)[:, -1] - np.partition(
        ev.logits, -2, axis=1
    )[:, -2]
    scores = scores - 1e-6 * margin
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

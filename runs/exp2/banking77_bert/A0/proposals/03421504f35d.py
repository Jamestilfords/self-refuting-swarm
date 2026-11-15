import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "An in-scope query should be supported by nearby examples from the intent predicted by its logits, "
    "not merely by nearby examples from other intents. The ratio of predicted-intent neighborhood distance "
    "to global neighborhood distance detects local representation-logit disagreement."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0089"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)

    k_global = min(20, len(x_train))
    global_dist = ctx.h.knn(x_train, x_eval, k=k_global, l2=False)

    pred = np.argmax(ev.logits, axis=1)
    class_dist = np.empty(len(x_eval), dtype=float)

    for c in np.unique(pred):
        eval_idx = np.flatnonzero(pred == c)
        train_idx = np.flatnonzero(tr.labels == c)
        if len(train_idx) == 0:
            class_dist[eval_idx] = np.max(global_dist) + 1.0
        else:
            k_class = min(10, len(train_idx))
            class_dist[eval_idx] = ctx.h.knn(
                x_train[train_idx], x_eval[eval_idx], k=k_class, l2=False
            )

    score_values = class_dist / (global_dist + 1e-8)
    return np.nan_to_num(score_values, nan=0.0, posinf=1e6, neginf=0.0)

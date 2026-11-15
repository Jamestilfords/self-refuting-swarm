import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Classifier-predicted intents should have recurring local neighborhoods in a balanced joint "
    "bottleneck-and-encoder representation. Class-conditional kNN distance exposes queries that "
    "are unsupported by the representation manifold of the intent selected by the classifier."
)
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0055", "m_0086"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def joint(x):
        a = ctx.h.l2(x.feats)
        b = ctx.h.l2(x.pooled)
        return ctx.h.l2(np.concatenate([a, b], axis=1))

    z_train = joint(tr)
    z_eval = joint(ev)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    n_train = z_train.shape[0]
    global_k = min(12, n_train)
    out = np.empty(z_eval.shape[0], dtype=float)

    for c in np.unique(eval_pred):
        query_idx = np.flatnonzero(eval_pred == c)
        support_idx = np.flatnonzero(train_pred == c)

        if support_idx.size < 4:
            support_idx = np.arange(n_train)

        k = min(12, support_idx.size)
        d_near = ctx.h.knn(z_train[support_idx], z_eval[query_idx], k=k, l2=True)

        if support_idx.size >= 24:
            d_mid = ctx.h.knn(z_train[support_idx], z_eval[query_idx], k=min(24, support_idx.size), l2=True)
            out[query_idx] = 0.7 * d_near + 0.3 * d_mid
        else:
            out[query_idx] = d_near

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=0.0)

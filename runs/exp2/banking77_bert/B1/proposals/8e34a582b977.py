import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "An in-scope query should be conformal to the feature distribution of the intent predicted by the classifier, "
    "whereas OOS queries can be close to some class yet atypical for their own predicted intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0025"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_dist = g.class_mahalanobis_sq(tr.pooled)
    eval_dist = g.class_mahalanobis_sq(ev.pooled)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = train_dist.shape[1]

    train_self = train_dist[np.arange(train_dist.shape[0]), train_pred]
    global_sorted = np.sort(train_self)
    scores = np.empty(eval_dist.shape[0], dtype=float)

    for c in range(n_classes):
        mask_eval = eval_pred == c
        if not np.any(mask_eval):
            continue

        mask_train = train_pred == c
        if np.sum(mask_train) >= 4:
            reference = np.sort(train_dist[mask_train, c])
        else:
            reference = global_sorted

        values = eval_dist[mask_eval, c]
        ranks = np.searchsorted(reference, values, side="right")
        scores[mask_eval] = ranks / float(reference.size + 1)

    return scores

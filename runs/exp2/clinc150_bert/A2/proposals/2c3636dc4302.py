import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query can be near an in-scope centroid yet be implausible for the class selected by the classifier. The predicted-class Mahalanobis tail probability measures disagreement between the classifier decision and class-conditional representation geometry."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0007"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_all = np.nan_to_num(
        g.class_mahalanobis_sq(tr.pooled),
        nan=0.0,
        posinf=1e12,
        neginf=0.0,
    )
    eval_all = np.nan_to_num(
        g.class_mahalanobis_sq(ev.pooled),
        nan=0.0,
        posinf=1e12,
        neginf=0.0,
    )

    n_classes = train_all.shape[1]
    train_pred = np.clip(np.argmax(tr.logits, axis=1), 0, n_classes - 1)
    eval_pred = np.clip(np.argmax(ev.logits, axis=1), 0, n_classes - 1)

    train_d = train_all[np.arange(train_all.shape[0]), train_pred]
    eval_d = eval_all[np.arange(eval_all.shape[0]), eval_pred]

    fallback = np.sort(train_d)
    scores = np.empty(eval_d.shape[0], dtype=float)

    for c in range(n_classes):
        eval_mask = eval_pred == c
        if not np.any(eval_mask):
            continue

        reference = np.sort(train_d[train_pred == c])
        if reference.size < 8:
            reference = fallback

        tail = np.searchsorted(
            reference, eval_d[eval_mask], side="right"
        ) / float(reference.size)

        q25, q75 = np.percentile(reference, [25.0, 75.0])
        scale = max(float(q75 - q25), 1e-6)
        median = float(np.median(reference))
        residual = np.tanh((eval_d[eval_mask] - median) / scale)

        scores[eval_mask] = tail + 1e-3 * residual

    return np.nan_to_num(scores, nan=0.0, posinf=1.001, neginf=-0.001)

import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope queries should align with the prototype of the intent selected by the classifier, while OOS queries can have an atypically weak predicted-intent cosine similarity. A disagreement penalty also flags cases where a different intent prototype is geometrically more compatible than the logit-selected intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0113", "m_0121"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    labels = tr.labels.astype(int)
    n_classes = tr.logits.shape[1]

    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    train_cos = x_train @ prototypes.T
    eval_cos = x_eval @ prototypes.T

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    medians = np.zeros(n_classes, dtype=float)
    scales = np.ones(n_classes, dtype=float)
    for c in range(n_classes):
        values = train_cos[train_pred == c, c]
        if values.size < 4:
            values = train_cos[labels == c, c]
        if values.size:
            medians[c] = np.median(values)
            q25, q75 = np.percentile(values, [25.0, 75.0])
            scales[c] = max(q75 - q25, 0.02)

    predicted_cos = eval_cos[np.arange(eval_cos.shape[0]), eval_pred]
    cosine_deficit = (medians[eval_pred] - predicted_cos) / scales[eval_pred]

    best_cos = np.max(eval_cos, axis=1)
    prototype_disagreement = np.maximum(best_cos - predicted_cos, 0.0)

    return cosine_deficit + 2.0 * prototype_disagreement

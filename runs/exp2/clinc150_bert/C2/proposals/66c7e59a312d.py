import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS queries may lie in regions where class-conditional feature distances are ambiguous, producing a small gap between the nearest and second-nearest class means. Class-conditional calibration on training predictions makes the ambiguity score comparable across predicted intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_two = np.partition(train_d, 1, axis=1)[:, :2]
    eval_two = np.partition(eval_d, 1, axis=1)[:, :2]

    train_margin = (train_two[:, 1] - train_two[:, 0]) / (
        train_two[:, 1] + train_two[:, 0] + 1e-8
    )
    eval_margin = (eval_two[:, 1] - eval_two[:, 0]) / (
        eval_two[:, 1] + eval_two[:, 0] + 1e-8
    )

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    means = np.full(n_classes, np.mean(train_margin))
    scales = np.full(n_classes, np.std(train_margin) + 1e-6)
    for c in range(n_classes):
        values = train_margin[train_pred == c]
        if values.size >= 2:
            means[c] = np.mean(values)
            scales[c] = np.std(values) + 1e-6

    return (means[eval_pred] - eval_margin) / scales[eval_pred]

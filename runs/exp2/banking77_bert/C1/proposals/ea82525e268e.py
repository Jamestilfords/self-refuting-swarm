import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "OOS queries may be assigned a confident predicted intent while remaining poorly aligned with that intent's normalized feature prototype. Class-conditional calibration of predicted-class cosine similarity accounts for differing within-class angular spread."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = tr.labels.astype(int)

    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]
    prototypes = np.zeros((n_classes, dim), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    train_pred = np.argmax(tr.logits, axis=1)
    train_cos = np.sum(x_train * prototypes[train_pred], axis=1)

    global_mean = float(np.mean(train_cos))
    global_std = float(np.std(train_cos))
    global_std = max(global_std, 1e-6)

    means = np.full(n_classes, global_mean, dtype=float)
    stds = np.full(n_classes, global_std, dtype=float)

    for c in range(n_classes):
        values = train_cos[train_pred == c]
        if values.size >= 5:
            class_mean = float(np.mean(values))
            class_std = float(np.std(values))
            weight = values.size / (values.size + 20.0)
            means[c] = weight * class_mean + (1.0 - weight) * global_mean
            stds[c] = max(
                weight * class_std + (1.0 - weight) * global_std,
                1e-6,
            )

    eval_pred = np.argmax(ev.logits, axis=1)
    eval_cos = np.sum(x_eval * prototypes[eval_pred], axis=1)

    standardized_deficit = (means[eval_pred] - eval_cos) / stds[eval_pred]

    all_cos = x_eval @ prototypes.T
    best_other = np.partition(all_cos, -2, axis=1)[:, -2]
    predicted_cos = all_cos[np.arange(all_cos.shape[0]), eval_pred]
    prototype_disagreement = best_other - predicted_cos

    return standardized_deficit + 0.2 * prototype_disagreement

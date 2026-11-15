import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "An in-scope representation should align with the prototype of the intent predicted by the classifier. Calibrating predicted-intent cosine deficits using training examples accounts for different prototype compactness across intents."
SOURCE = "https://proceedings.mlr.press/v267/muller25a.html"
PARENTS = []
MEMORY_USED = ["m_0129", "m_0137"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)

    n_classes = max(
        tr.logits.shape[1],
        ev.logits.shape[1],
        int(np.max(tr.labels)) + 1,
    )
    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    np.add.at(prototypes, tr.labels, x_train)
    counts = np.bincount(tr.labels, minlength=n_classes).astype(float)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_cos = np.sum(x_train * prototypes[train_pred], axis=1)
    train_deficit = 1.0 - train_cos

    sums = np.bincount(train_pred, weights=train_deficit, minlength=n_classes)
    sums_sq = np.bincount(
        train_pred, weights=train_deficit * train_deficit, minlength=n_classes
    )
    pred_counts = np.bincount(train_pred, minlength=n_classes).astype(float)

    global_mean = float(np.mean(train_deficit))
    global_std = float(np.std(train_deficit))
    class_mean = sums / np.maximum(pred_counts, 1.0)
    class_var = sums_sq / np.maximum(pred_counts, 1.0) - class_mean * class_mean
    class_std = np.sqrt(np.maximum(class_var, 0.0))

    sparse = pred_counts < 5.0
    class_mean[sparse] = global_mean
    class_std[sparse] = global_std
    class_std = np.maximum(class_std, 1e-6)

    eval_cos = np.sum(x_eval * prototypes[eval_pred], axis=1)
    eval_deficit = 1.0 - eval_cos
    return (eval_deficit - class_mean[eval_pred]) / class_std[eval_pred]

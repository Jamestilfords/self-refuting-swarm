import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope queries should have unusually strong cosine alignment with a class prototype and a clear margin over competing prototypes. Class-conditional alignment calibration and prototype-margin ambiguity make this distinct from raw maximum cosine similarity."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = np.asarray(tr.labels)

    classes, inverse = np.unique(labels, return_inverse=True)
    n_classes = classes.size

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    np.add.at(prototypes, inverse, x_train)
    counts = np.bincount(inverse, minlength=n_classes).astype(float)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    train_cos = x_train @ prototypes.T
    eval_cos = x_eval @ prototypes.T

    own_cos = train_cos[np.arange(train_cos.shape[0]), inverse]
    class_mean = np.zeros(n_classes, dtype=float)
    class_std = np.zeros(n_classes, dtype=float)
    np.add.at(class_mean, inverse, own_cos)
    class_mean /= np.maximum(counts, 1.0)

    centered = own_cos - class_mean[inverse]
    np.add.at(class_std, inverse, centered * centered)
    class_std = np.sqrt(class_std / np.maximum(counts - 1.0, 1.0))
    class_std = np.maximum(class_std, 1e-4)

    train_top = np.max(train_cos, axis=1)
    if n_classes > 1:
        train_second = np.partition(train_cos, -2, axis=1)[:, -2]
        eval_second = np.partition(eval_cos, -2, axis=1)[:, -2]
    else:
        train_second = np.zeros_like(train_top)
        eval_second = np.zeros(eval_cos.shape[0], dtype=float)

    train_margin = train_top - train_second
    margin_mean = float(np.mean(train_margin))
    margin_std = max(float(np.std(train_margin)), 1e-4)

    eval_top_index = np.argmax(eval_cos, axis=1)
    eval_top = eval_cos[np.arange(eval_cos.shape[0]), eval_top_index]
    eval_margin = eval_top - eval_second

    alignment_deficit = (
        class_mean[eval_top_index] - eval_top
    ) / class_std[eval_top_index]
    ambiguity = (margin_mean - eval_margin) / margin_std

    return alignment_deficit + 0.35 * ambiguity

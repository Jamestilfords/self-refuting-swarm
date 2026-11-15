import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope examples should have class-typical cosine alignment with a normalized feature prototype, while OOS examples tend to be both weakly aligned and ambiguous among prototypes. Class-conditional calibration accounts for different within-class angular concentration."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = tr.labels.astype(int)

    n_classes = int(np.max(labels)) + 1
    dim = x_train.shape[1]
    prototypes = np.zeros((n_classes, dim), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    train_cos = x_train @ prototypes.T
    own_cos = train_cos[np.arange(train_cos.shape[0]), labels]

    class_mean = np.zeros(n_classes, dtype=float)
    class_scale = np.zeros(n_classes, dtype=float)
    np.add.at(class_mean, labels, own_cos)
    class_mean /= np.maximum(counts, 1.0)

    centered = own_cos - class_mean[labels]
    np.add.at(class_scale, labels, centered * centered)
    class_scale = np.sqrt(class_scale / np.maximum(counts, 1.0))
    global_scale = float(np.std(own_cos))
    class_scale = np.maximum(class_scale, max(global_scale * 0.1, 1e-4))

    train_top = np.max(train_cos, axis=1)
    if n_classes > 1:
        train_second = np.partition(train_cos, -2, axis=1)[:, -2]
    else:
        train_second = np.full(train_top.shape, -1.0)
    train_margin = train_top - train_second
    margin_mean = float(np.mean(train_margin))
    margin_scale = max(float(np.std(train_margin)), 1e-4)

    eval_cos = x_eval @ prototypes.T
    nearest = np.argmax(eval_cos, axis=1)
    top = eval_cos[np.arange(eval_cos.shape[0]), nearest]
    if n_classes > 1:
        second = np.partition(eval_cos, -2, axis=1)[:, -2]
    else:
        second = np.full(top.shape, -1.0)
    margin = top - second

    alignment_unusualness = (class_mean[nearest] - top) / class_scale[nearest]
    ambiguity = (margin_mean - margin) / margin_scale
    return alignment_unusualness + 0.2 * ambiguity

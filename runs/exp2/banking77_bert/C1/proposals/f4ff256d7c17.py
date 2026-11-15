import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Class-conditioned cosine calibration measures how atypical an item's best prototype alignment is relative to training examples of that prototype class. This preserves prototype-based semantic matching while avoiding a globally monotonic transform of maximum cosine similarity."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = tr.labels
    classes = np.unique(labels)

    prototypes = np.empty((classes.size, x_train.shape[1]), dtype=float)
    for i, cls in enumerate(classes):
        proto = x_train[labels == cls].mean(axis=0)
        prototypes[i] = proto / max(np.linalg.norm(proto), 1e-12)

    train_cos = x_train @ prototypes.T
    eval_cos = x_eval @ prototypes.T

    centers = np.empty(classes.size, dtype=float)
    scales = np.empty(classes.size, dtype=float)
    global_scale = np.std(train_cos.max(axis=1))
    global_scale = max(global_scale, 1e-6)

    for i, cls in enumerate(classes):
        own_cos = train_cos[labels == cls, i]
        centers[i] = np.median(own_cos)
        mad = np.median(np.abs(own_cos - centers[i]))
        scales[i] = max(1.4826 * mad, global_scale * 0.1, 1e-6)

    nearest = np.argmax(eval_cos, axis=1)
    best_cos = eval_cos[np.arange(eval_cos.shape[0]), nearest]
    return (centers[nearest] - best_cos) / scales[nearest]

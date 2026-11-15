import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "L2-normalized in-scope features should align closely with at least one class prototype, while OOS queries are expected to have lower maximum cosine similarity to all learned class directions."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels)

    train_norm = np.linalg.norm(x_train, axis=1, keepdims=True)
    train_unit = x_train / np.maximum(train_norm, 1e-12)

    classes = np.unique(labels)
    prototypes = np.empty((classes.size, train_unit.shape[1]), dtype=float)
    for i, cls in enumerate(classes):
        proto = train_unit[labels == cls].mean(axis=0)
        prototypes[i] = proto / max(np.linalg.norm(proto), 1e-12)

    eval_norm = np.linalg.norm(x_eval, axis=1, keepdims=True)
    eval_unit = x_eval / np.maximum(eval_norm, 1e-12)
    max_cosine = np.max(eval_unit @ prototypes.T, axis=1)
    return 1.0 - np.clip(max_cosine, -1.0, 1.0)

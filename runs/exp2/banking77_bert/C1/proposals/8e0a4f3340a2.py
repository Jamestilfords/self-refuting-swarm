import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "In-scope examples should have unusually high cosine alignment with their nearest normalized class prototype compared with training examples assigned to that prototype. A small gap between the best and second-best prototypes additionally flags ambiguous directions between known intents."
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
    for j, c in enumerate(classes):
        proto = x_train[labels == c].mean(axis=0)
        prototypes[j] = proto / max(np.linalg.norm(proto), 1e-12)

    train_cos = x_train @ prototypes.T
    train_best_idx = np.argmax(train_cos, axis=1)
    train_best_cos = train_cos[np.arange(train_cos.shape[0]), train_best_idx]

    reference = []
    global_reference = np.sort(train_best_cos)
    for j in range(classes.size):
        values = train_best_cos[train_best_idx == j]
        reference.append(np.sort(values) if values.size else global_reference)

    eval_cos = x_eval @ prototypes.T
    eval_best_idx = np.argmax(eval_cos, axis=1)
    eval_best_cos = eval_cos[np.arange(eval_cos.shape[0]), eval_best_idx]

    if prototypes.shape[0] > 1:
        top_two = np.partition(eval_cos, -2, axis=1)[:, -2:]
        margin = top_two[:, 1] - top_two[:, 0]
    else:
        margin = np.ones(eval_best_cos.shape[0])

    tail = np.empty(eval_best_cos.shape[0], dtype=float)
    for j in range(classes.size):
        mask = eval_best_idx == j
        if np.any(mask):
            values = reference[j]
            ranks = np.searchsorted(values, eval_best_cos[mask], side="right")
            tail[mask] = 1.0 - (ranks + 0.5) / (values.size + 1.0)

    ambiguity = 1.0 - np.clip(margin / 2.0, 0.0, 1.0)
    return tail + 0.35 * ambiguity

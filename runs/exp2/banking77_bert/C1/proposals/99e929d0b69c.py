import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Class-specific cosine calibration measures whether an evaluation feature is unusually far from its nearest normalized class prototype relative to that class's in-scope spread. A small prototype margin additionally identifies queries that fall ambiguously between intent regions."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    labels = tr.labels.astype(int)
    n_classes = tr.logits.shape[1]

    train_norm = np.linalg.norm(tr.feats, axis=1, keepdims=True)
    eval_norm = np.linalg.norm(ev.feats, axis=1, keepdims=True)
    x_train = tr.feats / np.maximum(train_norm, 1e-12)
    x_eval = ev.feats / np.maximum(eval_norm, 1e-12)

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)

    own_similarity = np.sum(x_train * prototypes[labels], axis=1)
    global_mean = float(np.mean(own_similarity))
    global_std = float(np.std(own_similarity))
    global_std = max(global_std, 1e-4)

    class_mean = np.full(n_classes, global_mean, dtype=float)
    class_std = np.full(n_classes, global_std, dtype=float)
    for c in range(n_classes):
        mask = labels == c
        if np.any(mask):
            values = own_similarity[mask]
            class_mean[c] = (values.sum() + 4.0 * global_mean) / (values.size + 4.0)
            centered = values - class_mean[c]
            class_std[c] = max(
                np.sqrt((np.sum(centered * centered) + 4.0 * global_std ** 2)
                        / (values.size + 4.0)),
                1e-4,
            )

    similarity = x_eval @ prototypes.T
    best_class = np.argmax(similarity, axis=1)
    best = similarity[np.arange(similarity.shape[0]), best_class]

    if n_classes > 1:
        second = np.partition(similarity, -2, axis=1)[:, -2]
        ambiguity = 1.0 - (best - second)
    else:
        ambiguity = np.zeros_like(best)

    calibrated_distance = (class_mean[best_class] - best) / class_std[best_class]
    return calibrated_distance + 0.35 * ambiguity

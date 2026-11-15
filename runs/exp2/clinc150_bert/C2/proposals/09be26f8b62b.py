import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope examples should have cosine similarity to their nearest normalized class prototype consistent with that class's training dispersion. Class-conditional calibration makes unusually weak prototype alignment more anomalous than raw maximum cosine alone."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = np.asarray(tr.labels)
    classes, inverse = np.unique(labels, return_inverse=True)
    n_classes = len(classes)
    dim = x_train.shape[1]

    sums = np.zeros((n_classes, dim), dtype=float)
    counts = np.bincount(inverse, minlength=n_classes).astype(float)
    np.add.at(sums, inverse, x_train)

    prototypes = sums / np.maximum(counts[:, None], 1.0)
    proto_norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes = prototypes / np.maximum(proto_norms, 1e-12)

    loo_sums = sums[inverse] - x_train
    loo_counts = np.maximum(counts[inverse] - 1.0, 1.0)
    loo_proto = loo_sums / loo_counts[:, None]
    loo_norms = np.linalg.norm(loo_proto, axis=1)
    loo_proto = loo_proto / np.maximum(loo_norms[:, None], 1e-12)
    train_cos = np.sum(x_train * loo_proto, axis=1)

    class_mean = np.zeros(n_classes, dtype=float)
    class_std = np.zeros(n_classes, dtype=float)
    global_std = max(float(np.std(train_cos)), 1e-3)
    for j in range(n_classes):
        values = train_cos[inverse == j]
        class_mean[j] = float(np.mean(values))
        class_std[j] = max(float(np.std(values)), 0.1 * global_std, 1e-3)

    similarities = x_eval @ prototypes.T
    nearest = np.argmax(similarities, axis=1)
    best_cos = similarities[np.arange(len(x_eval)), nearest]
    scores = (class_mean[nearest] - best_cos) / class_std[nearest]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

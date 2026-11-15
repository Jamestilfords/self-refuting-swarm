import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class prototypes in normalized bottleneck space provide a cosine-based notion of semantic alignment. Calibrating maximum prototype similarity by the predicted prototype's in-scope similarity distribution accounts for classes with naturally different feature compactness."
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

    train_sim = x_train @ prototypes.T
    train_pred = np.argmax(train_sim, axis=1)
    train_best = train_sim[np.arange(train_sim.shape[0]), train_pred]

    means = np.zeros(n_classes, dtype=float)
    scales = np.ones(n_classes, dtype=float)
    for c in range(n_classes):
        values = train_best[train_pred == c]
        if values.size:
            means[c] = np.mean(values)
            scales[c] = max(np.std(values), 1e-3)
        else:
            means[c] = np.mean(train_best)
            scales[c] = max(np.std(train_best), 1e-3)

    eval_sim = x_eval @ prototypes.T
    eval_pred = np.argmax(eval_sim, axis=1)
    eval_best = eval_sim[np.arange(eval_sim.shape[0]), eval_pred]
    return (means[eval_pred] - eval_best) / scales[eval_pred]

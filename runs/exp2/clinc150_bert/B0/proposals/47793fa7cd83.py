import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "An in-scope query should be angularly consistent with the prototype of the intent predicted by its logits. Class-conditional calibration accounts for intents whose normalized features are naturally more or less concentrated, making unusually low predicted-class cosine similarity an OOS signal."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_x = np.asarray(tr.feats, dtype=np.float64)
    eval_x = np.asarray(ev.feats, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)

    train_x = train_x / np.maximum(np.linalg.norm(train_x, axis=1, keepdims=True), 1e-12)
    eval_x = eval_x / np.maximum(np.linalg.norm(eval_x, axis=1, keepdims=True), 1e-12)

    n_classes = tr.logits.shape[1]
    dim = train_x.shape[1]
    prototypes = np.zeros((n_classes, dim), dtype=np.float64)
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)

    np.add.at(prototypes, labels, train_x)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)

    own_cos = np.sum(train_x * prototypes[labels], axis=1)
    class_mean = np.zeros(n_classes, dtype=np.float64)
    class_sq_mean = np.zeros(n_classes, dtype=np.float64)
    np.add.at(class_mean, labels, own_cos)
    np.add.at(class_sq_mean, labels, own_cos * own_cos)
    class_mean /= np.maximum(counts, 1.0)
    class_var = class_sq_mean / np.maximum(counts, 1.0) - class_mean * class_mean
    class_std = np.sqrt(np.maximum(class_var, 1e-6))

    pred = np.argmax(np.asarray(ev.logits), axis=1)
    pred_cos = np.sum(eval_x * prototypes[pred], axis=1)

    standardized_deficit = (class_mean[pred] - pred_cos) / class_std[pred]
    raw_deficit = 1.0 - pred_cos
    scores = standardized_deficit + 0.01 * raw_deficit

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

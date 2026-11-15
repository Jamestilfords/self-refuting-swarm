import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "In-scope queries should align closely with the l2-normalized prototype of the class selected by the classifier. Low predicted-class cosine similarity, calibrated against training examples routed to that class, indicates semantic mismatch with known intent structure."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0099"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]

    train_norm = np.linalg.norm(x_train, axis=1, keepdims=True)
    eval_norm = np.linalg.norm(x_eval, axis=1, keepdims=True)
    x_train = x_train / np.maximum(train_norm, 1e-12)
    x_eval = x_eval / np.maximum(eval_norm, 1e-12)

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_cos = np.sum(x_train * prototypes[train_pred], axis=1)
    eval_cos = np.sum(x_eval * prototypes[eval_pred], axis=1)

    sums = np.bincount(train_pred, weights=train_cos, minlength=n_classes)
    pred_counts = np.bincount(train_pred, minlength=n_classes).astype(float)
    global_mean = float(np.mean(train_cos))
    means = sums / np.maximum(pred_counts, 1.0)
    means = np.where(pred_counts > 1.0, means, global_mean)

    centered = train_cos - means[train_pred]
    sq_sums = np.bincount(train_pred, weights=centered * centered, minlength=n_classes)
    global_std = float(np.std(train_cos))
    stds = np.sqrt(sq_sums / np.maximum(pred_counts - 1.0, 1.0))
    fallback_std = max(global_std, 1e-6)
    stds = np.where((pred_counts > 1.0) & (stds > 1e-6), stds, fallback_std)

    return (means[eval_pred] - eval_cos) / stds[eval_pred]

import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Cosine alignment with a known-intent prototype measures whether a bottleneck representation points in a direction supported by that intent. Calibrating the alignment relative to the predicted class's training distribution accounts for intent-specific prototype dispersion and avoids a globally equivalent raw-cosine ranking."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0075"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    labels = tr.labels.astype(int)
    n_classes = tr.logits.shape[1]

    train_norm = np.linalg.norm(tr.feats, axis=1, keepdims=True)
    eval_norm = np.linalg.norm(ev.feats, axis=1, keepdims=True)
    x_train = tr.feats / np.maximum(train_norm, 1e-12)
    x_eval = ev.feats / np.maximum(eval_norm, 1e-12)

    prototypes = np.zeros((n_classes, tr.feats.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes /= np.maximum(
        np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12
    )

    train_pred = np.argmax(tr.logits, axis=1)
    train_sim = np.sum(x_train * prototypes[train_pred], axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    eval_sim = np.sum(x_eval * prototypes[eval_pred], axis=1)

    sums = np.bincount(train_pred, weights=train_sim, minlength=n_classes)
    pred_counts = np.bincount(train_pred, minlength=n_classes).astype(float)
    global_mean = float(np.mean(train_sim))
    means = sums / np.maximum(pred_counts, 1.0)
    means = np.where(pred_counts > 1, means, global_mean)

    centered = train_sim - means[train_pred]
    sq_sums = np.bincount(
        train_pred, weights=centered * centered, minlength=n_classes
    )
    global_scale = float(np.std(train_sim))
    scales = np.sqrt(sq_sums / np.maximum(pred_counts - 1.0, 1.0))
    scales = np.where(pred_counts > 2, scales, global_scale)
    scales = np.maximum(scales, 1e-6)

    return (means[eval_pred] - eval_sim) / scales[eval_pred]

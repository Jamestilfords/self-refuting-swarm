import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "L2-normalized in-scope features should have unusually high cosine alignment with the prototype "
    "of the classifier-predicted intent. Class-conditional calibration makes low alignment relative "
    "to that intent's own training distribution an OOS signal rather than merely favoring broad classes."
)
SOURCE = "https://proceedings.mlr.press/v267/muller25a.html"
PARENTS = []
MEMORY_USED = ["m_0128", "m_0129"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)

    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]
    sums = np.zeros((n_classes, dim), dtype=np.float64)
    counts = np.bincount(tr.labels, minlength=n_classes).astype(np.float64)
    np.add.at(sums, tr.labels, x_train)

    prototypes = sums / np.maximum(counts[:, None], 1.0)
    prototypes /= np.maximum(
        np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12
    )

    train_cos = np.sum(x_train * prototypes[tr.labels], axis=1)
    class_sum = np.bincount(
        tr.labels, weights=train_cos, minlength=n_classes
    ).astype(np.float64)
    class_mean = class_sum / np.maximum(counts, 1.0)

    centered = train_cos - class_mean[tr.labels]
    class_var = np.bincount(
        tr.labels, weights=centered * centered, minlength=n_classes
    ).astype(np.float64) / np.maximum(counts, 1.0)
    class_std = np.sqrt(np.maximum(class_var, 1e-8))

    predicted = np.argmax(ev.logits, axis=1)
    eval_cos = np.sum(x_eval * prototypes[predicted], axis=1)
    alignment_z = (eval_cos - class_mean[predicted]) / class_std[predicted]

    return -np.clip(alignment_z, -20.0, 20.0)

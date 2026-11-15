import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Class-conditioned cosine support measures whether an item aligns with the angular direction "
    "of the intent predicted by the classifier, while train-fitted per-intent calibration detects unusually weak support."
)
SOURCE = "https://github.com/pradhyumnaag30/CLINC150"
PARENTS = []
MEMORY_USED = ["m_0122", "m_0178"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    n_classes = int(np.max(tr.labels)) + 1

    counts = np.bincount(tr.labels, minlength=n_classes).astype(float)
    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    np.add.at(prototypes, tr.labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    train_similarity = np.sum(x_train * prototypes[tr.labels], axis=1)
    class_mean = np.zeros(n_classes, dtype=float)
    np.add.at(class_mean, tr.labels, train_similarity)
    class_mean /= np.maximum(counts, 1.0)

    centered_sq = (train_similarity - class_mean[tr.labels]) ** 2
    class_var = np.zeros(n_classes, dtype=float)
    np.add.at(class_var, tr.labels, centered_sq)
    class_std = np.sqrt(class_var / np.maximum(counts - 1.0, 1.0))
    class_std = np.maximum(class_std, 1e-4)

    predicted = np.argmax(ev.logits, axis=1)
    predicted = np.clip(predicted, 0, n_classes - 1)
    eval_similarity = np.sum(x_eval * prototypes[predicted], axis=1)

    return (class_mean[predicted] - eval_similarity) / class_std[predicted]

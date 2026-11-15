import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "L2-normalized bottleneck features from known intents should align with their class prototype. "
    "Low cosine similarity to the classifier-predicted prototype indicates unsupported intent geometry."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0119"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    predicted = np.argmax(ev.logits, axis=1)
    similarity = np.sum(x_eval * prototypes[predicted], axis=1)

    train_predicted = np.argmax(tr.logits, axis=1)
    train_similarity = np.sum(x_train * prototypes[train_predicted], axis=1)
    class_mean = np.zeros(n_classes, dtype=float)
    np.add.at(class_mean, train_predicted, train_similarity)
    pred_counts = np.bincount(train_predicted, minlength=n_classes).astype(float)
    class_mean /= np.maximum(pred_counts, 1.0)

    scores = class_mean[predicted] - similarity
    return np.asarray(scores, dtype=float)

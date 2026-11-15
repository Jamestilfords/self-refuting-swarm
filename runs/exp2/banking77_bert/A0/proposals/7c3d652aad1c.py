import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "An in-scope query should align with the prototype of the class favored by the classifier. Probability-weighted prototype mismatch penalizes both weak directional support and disagreement between the classifier distribution and normalized feature geometry."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0009", "m_0013"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)

    n_classes = tr.logits.shape[1]
    labels = tr.labels.astype(int)
    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)

    np.add.at(prototypes, labels, x_train)
    global_prototype = np.mean(x_train, axis=0)
    missing = counts == 0
    prototypes[missing] = global_prototype
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    similarities = x_eval @ prototypes.T
    probabilities = ctx.h.softmax(ev.logits)
    expected_similarity = np.sum(probabilities * similarities, axis=1)

    return 1.0 - expected_similarity

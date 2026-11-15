import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A concentration-weighted mixture of cosine prototype similarities measures angular support from "
    "all known intents while giving tighter intent clusters more discriminative influence."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0009", "m_0020"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = tr.labels.astype(int)

    n_classes = int(np.max(labels)) + 1
    counts = np.bincount(labels, minlength=n_classes).astype(float)

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    own_similarity = np.sum(x_train * prototypes[labels], axis=1)
    class_sum = np.bincount(
        labels, weights=own_similarity, minlength=n_classes
    ).astype(float)
    class_mean = class_sum / np.maximum(counts, 1.0)

    # More concentrated intents receive sharper angular matching, but clipping
    # prevents a nearly degenerate class from dominating the mixture.
    temperature = np.clip(1.0 - class_mean, 0.04, 0.35)
    log_prior = np.log(np.maximum(counts / np.sum(counts), 1e-12))

    similarities = x_eval @ prototypes.T
    scaled = similarities / temperature[None, :] + log_prior[None, :]
    max_scaled = np.max(scaled, axis=1, keepdims=True)
    log_support = (
        max_scaled[:, 0]
        + np.log(np.sum(np.exp(scaled - max_scaled), axis=1))
    )

    return -log_support

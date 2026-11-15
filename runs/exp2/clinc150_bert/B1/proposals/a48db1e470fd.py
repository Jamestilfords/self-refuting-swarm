import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope examples should have their logit-routed class prototype among the strongest "
    "cosine matches in normalized bottleneck space. A large cosine prototype-consistency "
    "deficit flags queries whose semantic geometry disagrees with the classifier route."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0141", "m_0157"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(np.asarray(tr.feats, dtype=float))
    x_eval = ctx.h.l2(np.asarray(ev.feats, dtype=float))
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]

    sums = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(sums, labels, x_train)
    prototypes = sums / np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    affinities = x_eval @ prototypes.T
    routed = np.argmax(ev.logits, axis=1)
    routed_affinity = affinities[np.arange(affinities.shape[0]), routed]

    best_affinity = np.max(affinities, axis=1)
    mean_top = np.partition(affinities, -min(3, n_classes), axis=1)[
        :, -min(3, n_classes):
    ].mean(axis=1)

    deficit = best_affinity - routed_affinity
    weak_support = mean_top - routed_affinity
    return deficit + 0.5 * weak_support

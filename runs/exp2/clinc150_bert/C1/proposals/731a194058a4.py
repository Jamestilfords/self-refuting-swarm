import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope queries should align strongly with at least one normalized class prototype, while their alignment should also be typical for that class. Combining raw prototype mismatch with class-calibrated cosine atypicality distinguishes weakly aligned or unusually placed OOS queries."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = np.asarray(tr.labels)
    classes = np.unique(labels)

    prototypes = np.empty((classes.size, x_train.shape[1]), dtype=float)
    centers = np.empty(classes.size, dtype=float)
    scales = np.empty(classes.size, dtype=float)

    for j, cls in enumerate(classes):
        members = x_train[labels == cls]
        proto = members.mean(axis=0)
        proto_norm = np.linalg.norm(proto)
        if proto_norm > 1e-12:
            proto = proto / proto_norm
        prototypes[j] = proto

        member_sims = members @ proto
        centers[j] = np.mean(member_sims)
        scales[j] = max(np.std(member_sims), 0.02)

    similarities = x_eval @ prototypes.T
    raw_best = np.max(similarities, axis=1)
    calibrated_best = np.max(
        (similarities - centers[None, :]) / scales[None, :],
        axis=1,
    )

    return (1.0 - raw_best) - 0.35 * calibrated_best

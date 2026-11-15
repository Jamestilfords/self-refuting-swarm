import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "The classifier's predicted intent should have a closely aligned normalized feature prototype for in-scope queries. OOS queries can be forced toward a logit class while remaining angularly inconsistent with that class's training representation."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)

    k = tr.logits.shape[1]
    d = x_train.shape[1]
    prototypes = np.zeros((k, d), dtype=float)
    counts = np.zeros(k, dtype=float)

    valid = (tr.labels >= 0) & (tr.labels < k)
    np.add.at(prototypes, tr.labels[valid], x_train[valid])
    np.add.at(counts, tr.labels[valid], 1.0)

    global_proto = np.mean(x_train, axis=0, keepdims=True)
    global_proto = ctx.h.l2(global_proto)[0]
    missing = counts == 0
    prototypes[missing] = global_proto
    prototypes[~missing] /= counts[~missing, None]
    prototypes = ctx.h.l2(prototypes)

    predicted = np.argmax(ev.logits, axis=1)
    alignment = np.sum(x_eval * prototypes[predicted], axis=1)

    return 1.0 - alignment

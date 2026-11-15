import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS queries may have weak cosine support from the class prototypes favored by the classifier, rather than merely weak support from their single closest prototype. Posterior-weighted prototype alignment incorporates disagreement among plausible known intents and changes the ordering from a maximum-cosine baseline."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0081"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    k = tr.logits.shape[1]
    prototypes = np.empty((k, x_train.shape[1]), dtype=float)
    global_proto = x_train.mean(axis=0)
    global_proto /= max(np.linalg.norm(global_proto), 1e-12)

    for c in range(k):
        members = x_train[tr.labels == c]
        if members.shape[0] == 0:
            prototypes[c] = global_proto
        else:
            proto = members.mean(axis=0)
            prototypes[c] = proto / max(np.linalg.norm(proto), 1e-12)

    similarities = x_eval @ prototypes.T
    probs = ctx.h.softmax(ev.logits)
    expected_support = np.sum(probs * similarities, axis=1)

    sorted_similarities = np.sort(similarities, axis=1)
    prototype_gap = sorted_similarities[:, -1] - sorted_similarities[:, -2]
    return 1.0 - expected_support - 0.08 * prototype_gap

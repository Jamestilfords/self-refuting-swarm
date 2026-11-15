import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Rather than retaining only the nearest angular prototype, this score averages prototype agreement under the classifier's predicted class distribution. OOS queries tend to have low cosine agreement with the intents that their logits consider plausible."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0016"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)

    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]
    global_proto = x_train.mean(axis=0)
    global_proto /= max(np.linalg.norm(global_proto), 1e-12)

    prototypes = np.empty((n_classes, dim), dtype=float)
    for c in range(n_classes):
        members = x_train[tr.labels == c]
        if len(members) == 0:
            prototypes[c] = global_proto
        else:
            proto = members.mean(axis=0)
            prototypes[c] = proto / max(np.linalg.norm(proto), 1e-12)

    similarities = x_eval @ prototypes.T
    probabilities = ctx.h.softmax(ev.logits)
    expected_agreement = np.sum(probabilities * similarities, axis=1)

    return 1.0 - expected_agreement

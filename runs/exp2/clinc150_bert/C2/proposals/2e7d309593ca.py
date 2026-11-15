import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS queries may be assigned a confident logit class but still have weak cosine alignment with that predicted class's normalized feature prototype. Calibrating prototype conformity against training members of each predicted class accounts for class-specific feature concentration."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]

    global_proto = x_train.mean(axis=0)
    global_proto /= max(np.linalg.norm(global_proto), 1e-12)

    prototypes = np.empty((n_classes, dim), dtype=float)
    reference_scores = []
    for c in range(n_classes):
        mask = labels == c
        if np.any(mask):
            proto = x_train[mask].mean(axis=0)
            proto /= max(np.linalg.norm(proto), 1e-12)
            prototypes[c] = proto
            reference_scores.append(np.sort(x_train[mask] @ proto))
        else:
            prototypes[c] = global_proto
            reference_scores.append(np.array([-1.0, 1.0]))

    predicted = np.argmax(ev.logits, axis=1)
    cosine = np.sum(x_eval * prototypes[predicted], axis=1)

    scores = np.empty(cosine.shape[0], dtype=float)
    for c in range(n_classes):
        mask = predicted == c
        if np.any(mask):
            ref = reference_scores[c]
            lower_tail = np.searchsorted(ref, cosine[mask], side="right") / ref.size
            scores[mask] = 1.0 - lower_tail + 1e-3 * (1.0 - cosine[mask])

    return np.nan_to_num(scores, nan=1.0, posinf=1.0, neginf=0.0)

import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "EDL concentration vectors contain class-evidence shape information beyond total evidence or vacuity. "
    "An OOS query should have a normalized Dirichlet concentration profile that diverges from the train-fitted profile of its predicted intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    a_train = np.maximum(ctx.h.alpha(tr.logits), eps)
    a_eval = np.maximum(ctx.h.alpha(ev.logits), eps)

    p_train = a_train / np.sum(a_train, axis=1, keepdims=True)
    p_eval = a_eval / np.sum(a_eval, axis=1, keepdims=True)

    k = p_train.shape[1]
    global_profile = np.mean(p_train, axis=0)
    prototypes = np.empty((k, k), dtype=float)

    for c in range(k):
        members = p_train[tr.labels == c]
        if len(members) == 0:
            prototypes[c] = global_profile
        else:
            weight = len(members) / (len(members) + 8.0)
            prototypes[c] = weight * np.mean(members, axis=0) + (1.0 - weight) * global_profile

    prototypes = np.maximum(prototypes, eps)
    prototypes /= np.sum(prototypes, axis=1, keepdims=True)

    predicted = np.argmax(ev.logits, axis=1)
    q = prototypes[predicted]
    midpoint = 0.5 * (p_eval + q)

    js = 0.5 * np.sum(
        p_eval * (np.log(p_eval) - np.log(midpoint)),
        axis=1,
    ) + 0.5 * np.sum(
        q * (np.log(q) - np.log(midpoint)),
        axis=1,
    )
    return np.maximum(js, 0.0)

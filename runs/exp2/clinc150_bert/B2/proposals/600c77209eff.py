import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "EDL Dirichlet concentrations define a normalized evidence-allocation shape across intents. "
    "Queries whose predicted evidence profile diverges from the training profile of their assigned intent are atypical even when total evidence is high."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    alpha_train = np.asarray(ctx.h.alpha(tr.logits), dtype=float)
    alpha_eval = np.asarray(ctx.h.alpha(ev.logits), dtype=float)

    eps = 1e-8
    p_train = alpha_train / np.maximum(alpha_train.sum(axis=1, keepdims=True), eps)
    p_eval = alpha_eval / np.maximum(alpha_eval.sum(axis=1, keepdims=True), eps)

    n_classes = p_train.shape[1]
    prototypes = np.empty((n_classes, n_classes), dtype=float)
    global_profile = p_train.mean(axis=0)

    for c in range(n_classes):
        members = p_train[tr.labels == c]
        if len(members):
            prototypes[c] = members.mean(axis=0)
        else:
            prototypes[c] = global_profile

    prototypes = np.maximum(prototypes, eps)
    prototypes /= prototypes.sum(axis=1, keepdims=True)

    predicted = np.argmax(p_eval, axis=1)
    reference = prototypes[predicted]
    score_values = np.sum(
        p_eval * (np.log(np.maximum(p_eval, eps)) - np.log(reference)),
        axis=1,
    )
    return np.maximum(score_values, 0.0)

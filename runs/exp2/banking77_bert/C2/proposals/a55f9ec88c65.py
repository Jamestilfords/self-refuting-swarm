import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "Known intents tend to produce characteristic Dirichlet mean shapes across classes, whereas OOS inputs often have concentration proportions unlike every class-specific evidential prototype. The minimum Hellinger distance to trained class prototypes uses Dirichlet shape rather than total concentration, so it is not a monotone transform of energy or vacuity."
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

    n_classes = tr.logits.shape[1]
    prototypes = np.zeros((n_classes, p_train.shape[1]), dtype=float)
    counts = np.bincount(tr.labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, tr.labels, p_train)
    prototypes /= np.maximum(counts[:, None], 1.0)

    global_proto = np.mean(p_train, axis=0)
    missing = counts <= 0
    if np.any(missing):
        prototypes[missing] = global_proto

    root_eval = np.sqrt(p_eval)
    root_proto = np.sqrt(np.maximum(prototypes, eps))
    dist_sq = (
        np.sum(root_eval * root_eval, axis=1, keepdims=True)
        + np.sum(root_proto * root_proto, axis=1)[None, :]
        - 2.0 * root_eval.dot(root_proto.T)
    )
    return np.maximum(np.min(dist_sq, axis=1), 0.0)

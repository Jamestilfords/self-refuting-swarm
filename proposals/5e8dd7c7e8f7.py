import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "Normalized Dirichlet concentrations define a probability simplex whose Fisher-Rao geometry "
    "measures changes in evidential allocation independently of total evidence or vacuity. "
    "In-scope examples should have class-specific concentration shapes close to their learned "
    "Fisher-Rao prototype, while OOS inputs can induce atypical evidence allocations."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0104"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def simplex(alpha):
        alpha = np.nan_to_num(alpha, nan=1.0, posinf=1e12, neginf=1e-12)
        alpha = np.maximum(alpha, 1e-12)
        return alpha / np.maximum(alpha.sum(axis=1, keepdims=True), 1e-12)

    p_train = simplex(ctx.h.alpha(tr.logits))
    p_eval = simplex(ctx.h.alpha(ev.logits))

    s_train = np.sqrt(p_train)
    s_eval = np.sqrt(p_eval)
    y_train = np.argmax(tr.logits, axis=1)
    y_eval = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_proto = s_train.mean(axis=0)
    global_proto /= np.maximum(np.linalg.norm(global_proto), 1e-12)
    global_dist = 2.0 * np.arccos(
        np.clip(s_train @ global_proto, -1.0, 1.0)
    )
    global_mean = float(global_dist.mean())
    global_std = float(max(global_dist.std(), 1e-6))

    prototypes = np.empty((n_classes, tr.logits.shape[1]), dtype=float)
    means = np.empty(n_classes, dtype=float)
    stds = np.empty(n_classes, dtype=float)

    for c in range(n_classes):
        mask = y_train == c
        if mask.sum() < 2:
            prototypes[c] = global_proto
            means[c] = global_mean
            stds[c] = global_std
            continue

        proto = s_train[mask].mean(axis=0)
        proto /= np.maximum(np.linalg.norm(proto), 1e-12)
        dist = 2.0 * np.arccos(np.clip(s_train[mask] @ proto, -1.0, 1.0))

        prototypes[c] = proto
        means[c] = dist.mean()
        stds[c] = max(dist.std(), 1e-6)

    dist_eval = 2.0 * np.arccos(
        np.clip(np.sum(s_eval * prototypes[y_eval], axis=1), -1.0, 1.0)
    )
    return (dist_eval - means[y_eval]) / stds[y_eval]

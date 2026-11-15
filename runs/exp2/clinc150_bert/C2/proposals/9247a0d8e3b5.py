import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "This scores whether an item's normalized Dirichlet concentration shape is typical of its "
    "predicted in-scope class, rather than using total evidence or energy. OOS queries can have "
    "unusual allocations of evidence across competing intents even when their total concentration is high."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    alpha_tr = np.clip(np.asarray(ctx.h.alpha(tr.logits), dtype=float), 1e-10, 1e10)
    alpha_ev = np.clip(np.asarray(ctx.h.alpha(ev.logits), dtype=float), 1e-10, 1e10)

    shape_tr = np.log(alpha_tr) - np.log(np.sum(alpha_tr, axis=1, keepdims=True))
    shape_ev = np.log(alpha_ev) - np.log(np.sum(alpha_ev, axis=1, keepdims=True))

    n_classes = alpha_tr.shape[1]
    global_mean = np.mean(shape_tr, axis=0)
    global_var = np.var(shape_tr, axis=0) + 1e-6

    means = np.empty((n_classes, n_classes), dtype=float)
    variances = np.empty((n_classes, n_classes), dtype=float)

    for c in range(n_classes):
        members = shape_tr[tr.labels == c]
        if members.shape[0] >= 2:
            class_mean = np.mean(members, axis=0)
            class_var = np.var(members, axis=0)
            weight = members.shape[0] / (members.shape[0] + 10.0)
            means[c] = weight * class_mean + (1.0 - weight) * global_mean
            variances[c] = weight * class_var + (1.0 - weight) * global_var + 1e-6
        elif members.shape[0] == 1:
            means[c] = 0.1 * members[0] + 0.9 * global_mean
            variances[c] = global_var
        else:
            means[c] = global_mean
            variances[c] = global_var

    predicted = np.argmax(ev.logits, axis=1)
    residual = shape_ev - means[predicted]
    shape_distance = np.mean(
        (residual * residual) / variances[predicted],
        axis=1,
    )

    probs = alpha_ev / np.sum(alpha_ev, axis=1, keepdims=True)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)

    return np.nan_to_num(shape_distance + 0.01 * entropy, nan=0.0, posinf=1e6, neginf=0.0)

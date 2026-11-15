import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The normalized Dirichlet concentration vector captures evidence allocation shape independently "
    "of total evidence (and thus vacuity). OOS inputs can exhibit concentration profiles atypical for "
    "the intent predicted by the classifier even when their overall evidence is high."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0074"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    a_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), eps)

    # Log proportions discard total concentration, retaining only Dirichlet shape.
    z_tr = np.log(a_tr / np.sum(a_tr, axis=1, keepdims=True) + eps)
    z_ev = np.log(a_ev / np.sum(a_ev, axis=1, keepdims=True) + eps)

    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]
    dim = z_tr.shape[1]

    global_mean = np.mean(z_tr, axis=0)
    global_var = np.var(z_tr, axis=0) + 1e-4

    means = np.empty((n_classes, dim), dtype=float)
    variances = np.empty((n_classes, dim), dtype=float)

    for c in range(n_classes):
        members = z_tr[labels == c]
        count = members.shape[0]
        if count == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            class_mean = np.mean(members, axis=0)
            # Shrink sparse intent estimates toward the global shape distribution.
            weight = count / (count + 10.0)
            means[c] = weight * class_mean + (1.0 - weight) * global_mean
            if count > 1:
                class_var = np.var(members, axis=0)
                variances[c] = weight * class_var + (1.0 - weight) * global_var
            else:
                variances[c] = global_var

    predicted = np.argmax(ev.logits, axis=1)
    delta = z_ev - means[predicted]
    return np.mean((delta * delta) / (variances[predicted] + 1e-4), axis=1)

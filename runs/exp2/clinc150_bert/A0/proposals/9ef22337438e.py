import numpy as np

FAMILY = "density"
ARM = "edl"
RATIONALE = (
    "The centered log-ratio shape of Dirichlet concentrations captures how evidential mass is distributed across intents, independent of total concentration or vacuity. "
    "Queries whose evidential allocation is atypical for their predicted intent receive a larger class-conditional shape distance."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0061"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    alpha_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    alpha_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    p_tr = alpha_tr / np.sum(alpha_tr, axis=1, keepdims=True)
    p_ev = alpha_ev / np.sum(alpha_ev, axis=1, keepdims=True)

    z_tr = np.log(np.maximum(p_tr, 1e-12))
    z_tr = z_tr - np.mean(z_tr, axis=1, keepdims=True)
    z_ev = np.log(np.maximum(p_ev, 1e-12))
    z_ev = z_ev - np.mean(z_ev, axis=1, keepdims=True)

    labels = tr.labels.astype(int)
    n_classes = tr.logits.shape[1]
    dim = z_tr.shape[1]

    global_mean = np.mean(z_tr, axis=0)
    global_var = np.var(z_tr, axis=0) + 1e-6
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
            if count > 1:
                class_var = np.var(members, axis=0)
            else:
                class_var = global_var
            weight = count / (count + 20.0)
            means[c] = weight * class_mean + (1.0 - weight) * global_mean
            variances[c] = weight * class_var + (1.0 - weight) * global_var + 1e-6

    predicted = np.argmax(alpha_ev, axis=1)
    delta = z_ev - means[predicted]
    return np.sum((delta * delta) / variances[predicted], axis=1)

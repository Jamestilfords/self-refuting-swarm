import numpy as np

FAMILY = "density"
ARM = "edl"
RATIONALE = (
    "After removing total Dirichlet evidence, in-scope queries retain class-specific "
    "relative concentration patterns across intents. A large predicted-class surprise "
    "in centered log-concentration space flags atypical evidential shapes without "
    "being a monotone transformation of energy or vacuity."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0073"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    # Centering log concentrations removes the total-evidence scale and retains
    # only the Dirichlet concentration profile.
    z_tr = np.log(a_tr)
    z_tr = z_tr - z_tr.mean(axis=1, keepdims=True)
    z_ev = np.log(a_ev)
    z_ev = z_ev - z_ev.mean(axis=1, keepdims=True)

    labels = tr.labels.astype(int)
    pred = np.argmax(ev.logits, axis=1).astype(int)
    n_classes = tr.logits.shape[1]
    dim = z_tr.shape[1]

    global_mean = z_tr.mean(axis=0)
    global_var = z_tr.var(axis=0) + 1e-5
    means = np.empty((n_classes, dim), dtype=float)
    variances = np.empty((n_classes, dim), dtype=float)

    for c in range(n_classes):
        members = z_tr[labels == c]
        if len(members) >= 2:
            mean_c = members.mean(axis=0)
            var_c = members.var(axis=0)
            shrink = len(members) / (len(members) + 20.0)
            means[c] = shrink * mean_c + (1.0 - shrink) * global_mean
            variances[c] = shrink * var_c + (1.0 - shrink) * global_var + 1e-5
        elif len(members) == 1:
            means[c] = 0.1 * members[0] + 0.9 * global_mean
            variances[c] = global_var
        else:
            means[c] = global_mean
            variances[c] = global_var

    delta = z_ev - means[pred]
    return np.sum((delta * delta) / variances[pred], axis=1)

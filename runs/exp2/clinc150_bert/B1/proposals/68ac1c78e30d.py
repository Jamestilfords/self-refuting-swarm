import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The centered log-Dirichlet concentration profile measures how evidence is distributed "
    "across intents while removing total concentration, so it is not a monotone transform of vacuity or energy. "
    "Class-conditional training-tail calibration flags atypical evidential shapes for the intent to which a query is routed."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    alpha_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    alpha_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    # Remove the common log-concentration component: retain only Dirichlet shape.
    z_tr = np.log(alpha_tr)
    z_tr = z_tr - z_tr.mean(axis=1, keepdims=True)
    z_ev = np.log(alpha_ev)
    z_ev = z_ev - z_ev.mean(axis=1, keepdims=True)

    route_tr = np.argmax(tr.logits, axis=1)
    route_ev = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]
    dim = z_tr.shape[1]

    global_mean = z_tr.mean(axis=0)
    global_var = z_tr.var(axis=0)
    positive_global = global_var[global_var > 1e-10]
    floor = float(np.median(positive_global)) * 1e-3 if positive_global.size else 1e-6
    floor = max(floor, 1e-8)
    global_var = np.maximum(global_var, floor)

    means = np.tile(global_mean, (n_classes, 1))
    variances = np.tile(global_var, (n_classes, 1))

    for c in range(n_classes):
        members = z_tr[route_tr == c]
        if members.shape[0] >= 2:
            means[c] = members.mean(axis=0)
            class_var = members.var(axis=0)
            # Mild shrinkage prevents small routed intents from producing unstable tails.
            weight = members.shape[0] / (members.shape[0] + 12.0)
            variances[c] = np.maximum(
                weight * class_var + (1.0 - weight) * global_var,
                floor,
            )

    tr_diff = z_tr - means[route_tr]
    tr_dist = np.mean((tr_diff * tr_diff) / variances[route_tr], axis=1)

    ev_diff = z_ev - means[route_ev]
    ev_dist = np.mean((ev_diff * ev_diff) / variances[route_ev], axis=1)

    global_tail_reference = np.sort(tr_dist)
    scores = np.empty(ev_dist.shape[0], dtype=float)

    for c in range(n_classes):
        mask = route_ev == c
        if not np.any(mask):
            continue
        reference = tr_dist[route_tr == c]
        if reference.size < 8:
            reference = global_tail_reference
        else:
            reference = np.sort(reference)
        ranks = np.searchsorted(reference, ev_dist[mask], side="right")
        scores[mask] = ranks / (reference.size + 1.0)

    return np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)

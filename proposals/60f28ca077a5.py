import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Per-intent diagonal variances identify coordinates that are normally stable, while the "
    "concentration of standardized residual energy detects sparse coordinate violations that a "
    "single Mahalanobis radius can hide. Training-fitted predicted-intent tail calibration makes "
    "both unusually large radius and unusual residual shape score as OOS."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0094"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    n_classes = tr.logits.shape[1]
    dim = x.shape[1]
    global_mean = x.mean(axis=0)
    global_var = x.var(axis=0) + 1e-6

    means = np.empty((n_classes, dim), dtype=np.float64)
    variances = np.empty((n_classes, dim), dtype=np.float64)

    for c in range(n_classes):
        xc = x[y == c]
        count = len(xc)
        if count == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = xc.mean(axis=0)
            local_var = xc.var(axis=0)
            shrink = count / (count + 25.0)
            variances[c] = (
                shrink * local_var + (1.0 - shrink) * global_var + 1e-6
            )

    train_pred = np.argmax(tr.logits, axis=1)
    train_diff = x - means[train_pred]
    train_resid = train_diff * train_diff / variances[train_pred]
    train_radius = train_resid.mean(axis=1)
    train_concentration = (
        train_resid.max(axis=1) / np.maximum(train_resid.sum(axis=1), 1e-12)
    )

    radius_refs = []
    concentration_refs = []
    for c in range(n_classes):
        mask = train_pred == c
        if mask.sum() >= 8:
            radius_refs.append(np.sort(train_radius[mask]))
            concentration_refs.append(np.sort(train_concentration[mask]))
        else:
            radius_refs.append(np.sort(train_radius))
            concentration_refs.append(np.sort(train_concentration))

    eval_pred = np.argmax(ev.logits, axis=1)
    eval_diff = z - means[eval_pred]
    eval_resid = eval_diff * eval_diff / variances[eval_pred]
    eval_radius = eval_resid.mean(axis=1)
    eval_concentration = (
        eval_resid.max(axis=1) / np.maximum(eval_resid.sum(axis=1), 1e-12)
    )

    out = np.empty(len(z), dtype=np.float64)
    for c in range(n_classes):
        mask = eval_pred == c
        if not np.any(mask):
            continue

        r_ref = radius_refs[c]
        q_ref = concentration_refs[c]
        nr = len(r_ref)
        nq = len(q_ref)

        r_rank = np.searchsorted(r_ref, eval_radius[mask], side="left")
        radius_tail = (nr - r_rank + 1.0) / (nr + 1.0)

        q_rank = np.searchsorted(q_ref, eval_concentration[mask], side="right")
        q_cdf = q_rank / (nq + 1.0)
        concentration_tail = np.maximum(
            2.0 * np.minimum(q_cdf, 1.0 - q_cdf),
            1.0 / (nq + 1.0),
        )

        out[mask] = -np.log(radius_tail) - np.log(concentration_tail)

    return np.maximum(out, 0.0)

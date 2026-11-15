import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A classifier-selected intent should be plausible under its own shrunk diagonal variance profile. "
    "OOS inputs often have both an unusually large standardized radius and residual energy concentrated in coordinates that are stable for the selected intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0066", "m_0236"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x_train.shape[1]
    eps = 1e-12

    global_mean = np.mean(x_train, axis=0)
    global_var = np.var(x_train, axis=0) + 1e-6

    means = np.empty((k, d), dtype=np.float64)
    variances = np.empty((k, d), dtype=np.float64)
    counts = np.bincount(labels, minlength=k).astype(np.float64)

    for c in range(k):
        members = x_train[labels == c]
        n = members.shape[0]
        if n == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            mean_c = np.mean(members, axis=0)
            var_c = np.mean((members - mean_c) ** 2, axis=0)
            shrink = n / (n + 24.0)
            means[c] = mean_c
            variances[c] = shrink * var_c + (1.0 - shrink) * global_var + 1e-6

    def components(x, predicted):
        delta = x - means[predicted]
        diag_radius = np.mean((delta * delta) / variances[predicted], axis=1)
        raw_radius = np.mean((delta * delta) / global_var, axis=1)
        constraint_focus = np.log((diag_radius + eps) / (raw_radius + eps))
        return diag_radius, constraint_focus

    train_pred = np.argmax(tr.logits, axis=1)
    train_radius, train_focus = components(x_train, train_pred)

    radius_refs = []
    focus_refs = []
    all_radius = np.sort(train_radius)
    all_focus = np.sort(train_focus)

    for c in range(k):
        mask = train_pred == c
        if np.sum(mask) >= 12:
            radius_refs.append(np.sort(train_radius[mask]))
            focus_refs.append(np.sort(train_focus[mask]))
        else:
            radius_refs.append(all_radius)
            focus_refs.append(all_focus)

    eval_pred = np.argmax(ev.logits, axis=1)
    eval_radius, eval_focus = components(x_eval, eval_pred)
    out = np.empty(x_eval.shape[0], dtype=np.float64)

    for c in range(k):
        mask = eval_pred == c
        if not np.any(mask):
            continue
        rref = radius_refs[c]
        fref = focus_refs[c]
        radius_tail = np.searchsorted(rref, eval_radius[mask], side="right") / float(rref.size)
        focus_tail = np.searchsorted(fref, eval_focus[mask], side="right") / float(fref.size)
        smooth = 1e-6 * (
            eval_radius[mask] / (1.0 + eval_radius[mask])
            + 0.5 * np.tanh(eval_focus[mask])
        )
        out[mask] = radius_tail + focus_tail + smooth

    return np.nan_to_num(out, nan=0.0, posinf=2.0, neginf=0.0)

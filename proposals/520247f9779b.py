import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A predicted intent should match its own coordinate-wise feature variability, not only a shared covariance. "
    "This score also detects unusual concentration of standardized residual energy, catching sparse violations that a total radius can hide."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0027"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = x_train.shape[1]

    global_mean = np.mean(x_train, axis=0)
    global_var = np.var(x_train, axis=0) + 1e-6
    means = np.empty((k, d), dtype=float)
    variances = np.empty((k, d), dtype=float)

    for c in range(k):
        xc = x_train[y == c]
        n_c = len(xc)
        if n_c == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = np.mean(xc, axis=0)
            class_var = np.var(xc, axis=0) + 1e-6
            shrink = n_c / (n_c + 20.0)
            variances[c] = shrink * class_var + (1.0 - shrink) * global_var

    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)

    def radius_and_concentration(x, pred):
        radii = np.empty(len(x), dtype=float)
        concentrations = np.empty(len(x), dtype=float)
        for c in np.unique(pred):
            idx = np.flatnonzero(pred == c)
            z2 = (x[idx] - means[c]) ** 2 / variances[c]
            total = np.sum(z2, axis=1) + 1e-12
            radii[idx] = total / d
            concentrations[idx] = np.sum(z2 * z2, axis=1) / (total * total)
        return radii, concentrations

    train_radius, train_conc = radius_and_concentration(x_train, pred_train)
    conc_center = np.empty(k, dtype=float)
    conc_scale = np.empty(k, dtype=float)

    global_center = np.median(train_conc)
    global_scale = np.median(np.abs(train_conc - global_center)) + 1e-8
    for c in range(k):
        vals = train_conc[pred_train == c]
        if len(vals) < 5:
            conc_center[c] = global_center
            conc_scale[c] = global_scale
        else:
            center = np.median(vals)
            conc_center[c] = center
            conc_scale[c] = np.median(np.abs(vals - center)) + 1e-8

    train_raw = train_radius + np.abs(
        train_conc - conc_center[pred_train]
    ) / conc_scale[pred_train]

    eval_radius, eval_conc = radius_and_concentration(x_eval, pred_eval)
    eval_raw = eval_radius + np.abs(
        eval_conc - conc_center[pred_eval]
    ) / conc_scale[pred_eval]

    global_raw_center = np.median(train_raw)
    global_raw_scale = np.median(np.abs(train_raw - global_raw_center)) + 1e-8
    scores = np.empty(len(x_eval), dtype=float)

    for c in np.unique(pred_eval):
        idx = np.flatnonzero(pred_eval == c)
        reference = np.sort(train_raw[pred_train == c])
        if len(reference) < 5:
            reference = np.sort(train_raw)
        tail = np.searchsorted(reference, eval_raw[idx], side="right") / len(reference)
        fine = np.clip(
            (eval_raw[idx] - global_raw_center) / global_raw_scale,
            -20.0,
            20.0,
        )
        scores[idx] = tail + 1e-5 * fine

    return np.nan_to_num(scores, nan=0.0, posinf=1.0002, neginf=-0.0002)

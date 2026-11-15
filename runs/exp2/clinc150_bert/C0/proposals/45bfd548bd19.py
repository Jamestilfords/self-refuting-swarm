import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "For confident predictions this retains the energy score, while examples beyond the training uncertainty range are ranked increasingly by a predicted-class diagonal feature distance. This can expose OOS inputs with uncertain logits that also fall outside the corresponding intent's bottleneck-feature support."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    y_train = np.asarray(tr.labels, dtype=int)

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]
    global_mean = np.mean(x_train, axis=0)
    global_var = np.var(x_train, axis=0) + 1e-6

    means = np.empty((n_classes, dim), dtype=float)
    variances = np.empty((n_classes, dim), dtype=float)

    for c in range(n_classes):
        members = x_train[y_train == c]
        if members.shape[0] < 2:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = np.mean(members, axis=0)
            class_var = np.var(members, axis=0)
            shrink = members.shape[0] / (members.shape[0] + 20.0)
            variances[c] = shrink * class_var + (1.0 - shrink) * global_var + 1e-6

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_delta = x_train - means[train_pred]
    eval_delta = x_eval - means[eval_pred]
    train_dist = np.mean(
        (train_delta * train_delta) / variances[train_pred], axis=1
    )
    eval_dist = np.mean(
        (eval_delta * eval_delta) / variances[eval_pred], axis=1
    )

    e_med = float(np.median(train_energy))
    e_iqr = float(np.percentile(train_energy, 75) - np.percentile(train_energy, 25))
    e_scale = max(e_iqr, 1e-6)

    d_med = float(np.median(train_dist))
    d_iqr = float(np.percentile(train_dist, 75) - np.percentile(train_dist, 25))
    d_scale = max(d_iqr, 1e-6)

    uncertainty_start = float(np.percentile(train_energy, 70))
    uncertainty_end = float(np.percentile(train_energy, 90))
    uncertainty_width = max(uncertainty_end - uncertainty_start, e_scale, 1e-6)

    gate = np.clip(
        (eval_energy - uncertainty_start) / uncertainty_width, 0.0, 1.0
    )
    energy_z = (eval_energy - e_med) / e_scale
    distance_z = (eval_dist - d_med) / d_scale

    scores = (1.0 - gate) * energy_z + gate * distance_z
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A classifier-selected intent should agree with a shrunk diagonal class-density model. "
    "OOS queries can create disagreement between the logit-selected intent and the most geometrically compatible intent, while distance from every class resolves remote ties."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0111"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = np.asarray(tr.feats, dtype=float)
    xev = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    n_classes = tr.logits.shape[1]
    dim = xtr.shape[1]
    global_mean = np.mean(xtr, axis=0)
    global_var = np.maximum(np.var(xtr, axis=0), 1e-6)

    means = np.empty((n_classes, dim), dtype=float)
    variances = np.empty((n_classes, dim), dtype=float)

    for c in range(n_classes):
        xc = xtr[y == c]
        count = xc.shape[0]
        if count == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = np.mean(xc, axis=0)
            if count > 1:
                class_var = np.var(xc, axis=0)
            else:
                class_var = global_var
            shrink = count / (count + 20.0)
            variances[c] = shrink * class_var + (1.0 - shrink) * global_var

    variances = np.maximum(variances, 1e-6)
    inv_var = 1.0 / variances
    linear = means * inv_var
    offset = np.sum(means * linear, axis=1)
    logdet = np.sum(np.log(variances), axis=1)

    def class_nll(x):
        square_term = (x * x) @ inv_var.T
        cross_term = x @ linear.T
        return 0.5 * (square_term - 2.0 * cross_term + offset + logdet)

    train_nll = class_nll(xtr)
    train_min = np.min(train_nll, axis=1)
    train_pred = np.argmax(tr.logits, axis=1)
    train_selected = train_nll[np.arange(xtr.shape[0]), train_pred]
    train_gap = train_selected - train_min

    gap_center = np.median(train_gap)
    gap_scale = np.median(np.abs(train_gap - gap_center)) + 1e-6
    dist_center = np.median(train_min)
    dist_scale = np.median(np.abs(train_min - dist_center)) + 1e-6

    eval_nll = class_nll(xev)
    eval_min = np.min(eval_nll, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    eval_selected = eval_nll[np.arange(xev.shape[0]), eval_pred]

    disagreement = (eval_selected - eval_min - gap_center) / gap_scale
    remoteness = (eval_min - dist_center) / dist_scale
    scores = disagreement + 0.25 * remoteness

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

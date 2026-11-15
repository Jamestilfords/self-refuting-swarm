import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Class-conditional centered logit profiles capture whether the full relative evidence pattern resembles training examples predicted as the same intent. OOS queries can have plausible energy or confidence while exhibiting an atypical distribution of non-winning logits."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr_logits = np.asarray(ctx.train.logits, dtype=float)
    ev_logits = np.asarray(ctx.eval.logits, dtype=float)

    tr_shape = tr_logits - np.mean(tr_logits, axis=1, keepdims=True)
    ev_shape = ev_logits - np.mean(ev_logits, axis=1, keepdims=True)

    tr_pred = np.argmax(tr_logits, axis=1)
    ev_pred = np.argmax(ev_logits, axis=1)
    n_classes = tr_logits.shape[1]

    counts = np.bincount(tr_pred, minlength=n_classes).astype(float)
    sums = np.zeros((n_classes, tr_logits.shape[1]), dtype=float)
    np.add.at(sums, tr_pred, tr_shape)

    global_mean = np.mean(tr_shape, axis=0)
    shrink = 12.0
    means = (sums + shrink * global_mean) / (counts[:, None] + shrink)

    residual = tr_shape - means[tr_pred]
    global_var = np.mean(residual * residual, axis=0)
    floor = max(float(np.mean(global_var)) * 1e-4, 1e-8)

    sq_sums = np.zeros((n_classes, tr_logits.shape[1]), dtype=float)
    np.add.at(sq_sums, tr_pred, residual * residual)
    variances = (sq_sums + shrink * global_var) / (counts[:, None] + shrink)
    variances = np.maximum(variances, floor)

    ev_residual = ev_shape - means[ev_pred]
    scores = np.mean((ev_residual * ev_residual) / variances[ev_pred], axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)

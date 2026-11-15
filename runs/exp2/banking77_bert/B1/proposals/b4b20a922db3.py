import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Different intents can have distinct feature-wise variability, so a query should be judged against the "
    "diagonal covariance of its classifier-predicted intent. Class-conditional empirical tail calibration prevents "
    "naturally diffuse intents from being systematically treated as out-of-scope."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    n_train = x_train.shape[0]
    pred_eval = np.argmax(np.asarray(ev.logits), axis=1)
    classes = np.unique(labels)

    global_mean = np.mean(x_train, axis=0)
    global_var = np.mean((x_train - global_mean) ** 2, axis=0)
    valid_var = global_var[np.isfinite(global_var) & (global_var > 0)]
    floor = float(np.median(valid_var) * 1e-5) if valid_var.size else 1e-8
    floor = max(floor, 1e-8)
    global_var = np.maximum(np.nan_to_num(global_var, nan=floor), floor)

    scores = np.empty(x_eval.shape[0], dtype=float)

    for cls in classes:
        train_mask = labels == cls
        eval_mask = pred_eval == cls
        xc = x_train[train_mask]
        count = xc.shape[0]

        mean = np.mean(xc, axis=0)
        class_var = np.mean((xc - mean) ** 2, axis=0)

        shrink = count / (count + 24.0)
        var = shrink * class_var + (1.0 - shrink) * global_var
        var = np.maximum(np.nan_to_num(var, nan=floor), floor)

        train_diff = xc - mean
        train_dist = np.sum((train_diff * train_diff) / var, axis=1)
        train_dist = np.sort(np.nan_to_num(train_dist, nan=0.0, posinf=1e12))

        if np.any(eval_mask):
            diff = x_eval[eval_mask] - mean
            dist = np.sum((diff * diff) / var, axis=1)
            dist = np.nan_to_num(dist, nan=0.0, posinf=1e12, neginf=0.0)
            ranks = np.searchsorted(train_dist, dist, side="right")
            scores[eval_mask] = ranks / float(count + 1)

    unseen_mask = ~np.isin(pred_eval, classes)
    if np.any(unseen_mask):
        diff = x_eval[unseen_mask] - global_mean
        scores[unseen_mask] = np.sum((diff * diff) / global_var, axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)

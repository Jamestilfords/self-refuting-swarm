import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Out-of-scope inputs often have a smaller gap between their most likely and second-most-likely intents. Class-conditional margin standardization accounts for intent classes whose in-scope examples naturally have different confidence margins."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def margins_and_preds(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        pred = np.argmax(logits, axis=1)
        return margin, pred

    train_margin, train_pred = margins_and_preds(tr.logits)
    eval_margin, eval_pred = margins_and_preds(ev.logits)

    k = tr.logits.shape[1]
    global_mean = float(np.mean(train_margin))
    global_var = float(np.mean((train_margin - global_mean) ** 2))
    global_scale = max(np.sqrt(global_var), 1e-6)

    counts = np.bincount(train_pred, minlength=k).astype(float)
    sums = np.bincount(train_pred, weights=train_margin, minlength=k)
    raw_means = sums / np.maximum(counts, 1.0)

    # Shrink rare predicted-class estimates toward the global train distribution.
    shrink = 20.0
    means = (counts * raw_means + shrink * global_mean) / (counts + shrink)

    centered_sq = (train_margin - means[train_pred]) ** 2
    var_sums = np.bincount(train_pred, weights=centered_sq, minlength=k)
    vars_ = (var_sums + shrink * global_scale ** 2) / (counts + shrink)
    scales = np.maximum(np.sqrt(vars_), 1e-6)

    return -(eval_margin - means[eval_pred]) / scales[eval_pred]

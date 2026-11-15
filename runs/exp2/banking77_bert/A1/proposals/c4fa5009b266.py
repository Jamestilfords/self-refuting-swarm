import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "OOS queries often have a smaller gap between their two most plausible intents, even when total logit energy remains high. Predicted-intent conditional calibration accounts for classes with inherently broader or narrower in-scope decision margins."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0059"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def top_two_margin(logits):
        part = np.partition(logits, -2, axis=1)
        return part[:, -1] - part[:, -2]

    train_margin = top_two_margin(tr.logits)
    eval_margin = top_two_margin(ev.logits)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_mean = float(np.mean(train_margin))
    global_var = float(np.mean((train_margin - global_mean) ** 2))
    global_scale = np.sqrt(max(global_var, 1e-8))

    counts = np.bincount(train_pred, minlength=n_classes).astype(float)
    sums = np.bincount(train_pred, weights=train_margin, minlength=n_classes)
    shrinkage = 8.0
    class_mean = (sums + shrinkage * global_mean) / (counts + shrinkage)

    score_values = (class_mean[eval_pred] - eval_margin) / global_scale
    return np.nan_to_num(score_values, nan=0.0, posinf=1e6, neginf=-1e6)

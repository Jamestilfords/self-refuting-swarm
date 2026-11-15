import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Out-of-scope inputs can produce logit vectors with an atypical overall magnitude even when their maximum class score is high. A robust two-sided deviation from the training logit-norm distribution captures this logit-shape mismatch."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    train_logits = np.asarray(ctx.train.logits, dtype=float)
    eval_logits = np.asarray(ctx.eval.logits, dtype=float)

    train_norm = np.sqrt(np.sum(train_logits * train_logits, axis=1))
    center = np.median(train_norm)
    mad = np.median(np.abs(train_norm - center))
    scale = max(1.4826 * mad, 1e-6)

    eval_norm = np.sqrt(np.sum(eval_logits * eval_logits, axis=1))
    scores = np.abs(eval_norm - center) / scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)

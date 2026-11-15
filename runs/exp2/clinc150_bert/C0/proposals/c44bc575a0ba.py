import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Fine-tuned in-scope examples typically occupy a characteristic range of logit magnitudes, while OOS queries often produce unusually weak or otherwise atypical activation strength. A robust deviation from the training logit-norm distribution detects both underconfident and anomalously overconfident logit shapes."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    train_norm = np.linalg.norm(np.asarray(ctx.train.logits, dtype=float), axis=1)
    eval_norm = np.linalg.norm(np.asarray(ctx.eval.logits, dtype=float), axis=1)

    center = np.median(train_norm)
    mad = np.median(np.abs(train_norm - center))
    scale = max(1.4826 * mad, 1e-8)

    return np.abs(eval_norm - center) / scale

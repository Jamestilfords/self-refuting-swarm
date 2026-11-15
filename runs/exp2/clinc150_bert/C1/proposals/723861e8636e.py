import numpy as np
from scipy.special import digamma

FAMILY = "other"
ARM = "edl"
RATIONALE = "Dirichlet mutual information measures disagreement between categorical predictions sampled from the evidential posterior, capturing concentration shape as well as total evidence. Class-conditional upper-tail calibration makes unusually distributionally uncertain queries score as OOS."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def mutual_information(logits):
        alpha = np.maximum(ctx.h.alpha(logits), 1e-8)
        total = np.sum(alpha, axis=1, keepdims=True)
        mean = alpha / total
        pred_entropy = -np.sum(mean * np.log(np.maximum(mean, 1e-12)), axis=1)
        expected_entropy = (
            digamma(total[:, 0] + 1.0)
            - np.sum(mean * digamma(alpha + 1.0), axis=1)
        )
        return np.maximum(pred_entropy - expected_entropy, 0.0)

    train_mi = mutual_information(tr.logits)
    eval_mi = mutual_information(ev.logits)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    scores = np.empty(len(eval_mi), dtype=float)

    global_sorted = np.sort(train_mi)
    for cls in np.unique(eval_pred):
        cls_train = train_mi[train_pred == cls]
        reference = np.sort(cls_train) if len(cls_train) >= 8 else global_sorted
        mask = eval_pred == cls
        scores[mask] = np.searchsorted(reference, eval_mi[mask], side="right") / (
            len(reference) + 1.0
        )

    return scores

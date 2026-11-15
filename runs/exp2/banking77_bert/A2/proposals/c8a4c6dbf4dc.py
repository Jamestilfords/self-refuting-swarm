import numpy as np
from scipy.special import digamma

FAMILY = "other"
ARM = "edl"
RATIONALE = "Dirichlet mutual information measures disagreement induced by the full concentration shape, separating epistemically ambiguous predictions from merely low-energy ones. Predicted-intent robust calibration flags evidence shapes atypical for the intent selected by the classifier."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0107"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def mutual_information(logits):
        alpha = np.maximum(ctx.h.alpha(logits), 1e-8)
        total = np.sum(alpha, axis=1)
        prob = alpha / total[:, None]
        pred_entropy = -np.sum(prob * np.log(np.maximum(prob, 1e-12)), axis=1)
        expected_entropy = (
            digamma(total + 1.0)
            - np.sum(
                prob * digamma(alpha + 1.0),
                axis=1,
            )
        )
        return pred_entropy - expected_entropy

    train_mi = mutual_information(tr.logits)
    eval_mi = mutual_information(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    k = tr.logits.shape[1]
    med = np.empty(k, dtype=float)
    scale = np.empty(k, dtype=float)
    global_med = np.median(train_mi)
    global_scale = np.median(np.abs(train_mi - global_med)) * 1.4826 + 1e-6

    for c in range(k):
        values = train_mi[train_pred == c]
        if values.size >= 5:
            med[c] = np.median(values)
            scale[c] = np.median(np.abs(values - med[c])) * 1.4826 + 1e-6
        else:
            med[c] = global_med
            scale[c] = global_scale

    scores = np.abs(eval_mi - med[eval_pred]) / scale[eval_pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)

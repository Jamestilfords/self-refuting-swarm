import numpy as np
from scipy.special import digamma

FAMILY = "other"
ARM = "edl"
RATIONALE = "Dirichlet mutual information measures disagreement induced by the full concentration shape, not just total evidence. Class-conditional calibration on training queries highlights inputs whose evidential posterior is unusually distributionally uncertain."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def mutual_information(alpha):
        a = np.maximum(np.asarray(alpha, dtype=float), 1e-8)
        a0 = np.sum(a, axis=1, keepdims=True)
        mean = a / a0
        entropy_mean = -np.sum(mean * np.log(np.maximum(mean, 1e-12)), axis=1)
        expected_entropy = digamma(a0[:, 0] + 1.0) - np.sum(
            mean * digamma(a + 1.0), axis=1
        )
        return entropy_mean - expected_entropy

    a_train = ctx.h.alpha(tr.logits)
    a_eval = ctx.h.alpha(ev.logits)
    mi_train = mutual_information(a_train)
    mi_eval = mutual_information(a_eval)

    pred_train = np.argmax(a_train, axis=1)
    pred_eval = np.argmax(a_eval, axis=1)
    k = a_train.shape[1]

    centers = np.empty(k, dtype=float)
    scales = np.empty(k, dtype=float)
    global_center = np.median(mi_train)
    global_scale = np.median(np.abs(mi_train - global_center)) * 1.4826
    global_scale = max(float(global_scale), 1e-6)

    for c in range(k):
        values = mi_train[pred_train == c]
        if values.size < 4:
            centers[c] = global_center
            scales[c] = global_scale
        else:
            center = np.median(values)
            scale = np.median(np.abs(values - center)) * 1.4826
            centers[c] = center
            scales[c] = max(float(scale), 1e-6)

    scores = (mi_eval - centers[pred_eval]) / scales[pred_eval]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

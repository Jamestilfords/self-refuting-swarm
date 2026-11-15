import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "The centered logit norm measures overall predictive sharpness independently of an additive logit offset. A two-sided, predicted-class-conditioned calibration flags both unusually diffuse and unusually overconfident logit shapes that are atypical of in-scope examples."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0011"]


def score(ctx):
    tr_logits = np.asarray(ctx.train.logits, dtype=float)
    ev_logits = np.asarray(ctx.eval.logits, dtype=float)

    tr_centered = tr_logits - np.mean(tr_logits, axis=1, keepdims=True)
    ev_centered = ev_logits - np.mean(ev_logits, axis=1, keepdims=True)
    tr_norm = np.sqrt(np.sum(tr_centered * tr_centered, axis=1))
    ev_norm = np.sqrt(np.sum(ev_centered * ev_centered, axis=1))

    tr_pred = np.argmax(tr_logits, axis=1)
    ev_pred = np.argmax(ev_logits, axis=1)
    n_classes = tr_logits.shape[1]

    global_med = np.median(tr_norm)
    global_scale = np.median(np.abs(tr_norm - global_med))
    global_scale = max(float(global_scale), 1e-6)

    medians = np.full(n_classes, global_med, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = tr_norm[tr_pred == c]
        if values.size >= 8:
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            medians[c] = med
            scales[c] = max(float(mad), 0.25 * global_scale, 1e-6)

    scores = np.abs(ev_norm - medians[ev_pred]) / scales[ev_pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)

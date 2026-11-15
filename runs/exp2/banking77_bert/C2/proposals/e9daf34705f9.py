import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Centered logit norm measures the overall separation of the predicted class from alternatives while removing arbitrary common logit offsets. Class-conditional calibration on training predictions makes unusually low separation score as more out-of-scope."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr_logits = np.asarray(ctx.train.logits, dtype=float)
    ev_logits = np.asarray(ctx.eval.logits, dtype=float)

    tr_centered = tr_logits - np.mean(tr_logits, axis=1, keepdims=True)
    ev_centered = ev_logits - np.mean(ev_logits, axis=1, keepdims=True)
    tr_norm = np.sqrt(np.mean(tr_centered * tr_centered, axis=1))
    ev_norm = np.sqrt(np.mean(ev_centered * ev_centered, axis=1))

    tr_pred = np.argmax(tr_logits, axis=1)
    ev_pred = np.argmax(ev_logits, axis=1)
    n_classes = tr_logits.shape[1]

    global_mean = float(np.mean(tr_norm))
    global_scale = float(np.std(tr_norm))
    if not np.isfinite(global_scale) or global_scale < 1e-6:
        global_scale = 1.0

    means = np.full(n_classes, global_mean, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = tr_norm[tr_pred == c]
        if values.size:
            mean = float(np.mean(values))
            scale = float(np.std(values))
            if np.isfinite(mean):
                means[c] = mean
            if np.isfinite(scale) and scale >= 1e-6:
                scales[c] = scale

    return -(ev_norm - means[ev_pred]) / scales[ev_pred]

import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "Temperature-softened logit entropy measures ambiguity across several plausible intents rather "
    "than only the winning class. OOS queries should retain broader tempered class uncertainty, and "
    "predicted-intent calibration accounts for classes with intrinsically different logit shapes."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    temperature = 2.5

    def tempered_entropy(logits):
        x = np.nan_to_num(
            logits / temperature, nan=0.0, posinf=1e4, neginf=-1e4
        )
        x = x - np.max(x, axis=1, keepdims=True)
        p = np.exp(np.clip(x, -700.0, 0.0))
        p /= np.maximum(np.sum(p, axis=1, keepdims=True), 1e-12)
        return -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=1)

    tr_logits = np.nan_to_num(tr.logits, nan=0.0, posinf=1e4, neginf=-1e4)
    ev_logits = np.nan_to_num(ev.logits, nan=0.0, posinf=1e4, neginf=-1e4)
    tr_pred = np.argmax(tr_logits, axis=1)
    ev_pred = np.argmax(ev_logits, axis=1)

    tr_score = tempered_entropy(tr_logits)
    ev_score = tempered_entropy(ev_logits)

    global_med = float(np.median(tr_score))
    global_mad = float(np.median(np.abs(tr_score - global_med))) * 1.4826
    global_scale = max(global_mad, 1e-6)

    n_classes = tr_logits.shape[1]
    centers = np.full(n_classes, global_med, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = tr_score[tr_pred == c]
        if values.size >= 12:
            center = float(np.median(values))
            scale = float(np.median(np.abs(values - center))) * 1.4826
            centers[c] = center
            scales[c] = max(scale, 1e-6)

    return np.clip((ev_score - centers[ev_pred]) / scales[ev_pred], -100.0, 100.0)

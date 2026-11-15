import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The collision concentration of normalized Dirichlet evidence measures whether mass is distributed "
    "across intents in a class-atypical way, independently of total evidence and vacuity. Two-sided "
    "predicted-intent calibration flags both unusually diffuse and unusually concentrated evidence shapes."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0258"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]

    def collision(logits):
        alpha = ctx.h.alpha(logits)
        alpha = np.nan_to_num(alpha, nan=1.0, posinf=1e12, neginf=1e-12)
        alpha = np.maximum(alpha, 1e-12)
        scaled = alpha / np.maximum(alpha.max(axis=1, keepdims=True), 1e-12)
        probs = scaled / np.maximum(scaled.sum(axis=1, keepdims=True), 1e-12)
        return np.sum(probs * probs, axis=1)

    train_collision = collision(tr.logits)
    eval_collision = collision(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_mean = float(np.mean(train_collision))
    global_var = float(np.var(train_collision))
    floor = max(global_var * 0.05, 1e-10)

    means = np.full(k, global_mean, dtype=float)
    variances = np.full(k, max(global_var, floor), dtype=float)

    for cls in range(k):
        values = train_collision[train_pred == cls]
        if values.size >= 4:
            weight = values.size / (values.size + 8.0)
            means[cls] = weight * float(np.mean(values)) + (1.0 - weight) * global_mean
            variances[cls] = max(
                weight * float(np.var(values)) + (1.0 - weight) * global_var,
                floor,
            )

    z = (eval_collision - means[eval_pred]) / np.sqrt(variances[eval_pred])
    return np.abs(z)

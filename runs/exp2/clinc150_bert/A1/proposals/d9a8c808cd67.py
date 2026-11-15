import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The normalized Dirichlet concentration pattern has an intent-specific shape that is distinct from total evidence or energy. "
    "Queries with unusually diffuse or unusually uneven concentration shapes relative to their predicted intent receive high scores."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0017"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape_features(logits):
        alpha = np.maximum(ctx.h.alpha(logits), 1e-8)
        prob = alpha / np.maximum(alpha.sum(axis=1, keepdims=True), 1e-8)
        entropy = -np.sum(prob * np.log(prob), axis=1)
        log_alpha = np.log(alpha)
        dispersion = np.mean(
            (log_alpha - log_alpha.mean(axis=1, keepdims=True)) ** 2,
            axis=1,
        )
        return np.column_stack((entropy, dispersion))

    train_shape = shape_features(tr.logits)
    eval_shape = shape_features(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_med = np.median(train_shape, axis=0)
    global_mad = np.median(np.abs(train_shape - global_med), axis=0)
    global_scale = np.maximum(1.4826 * global_mad, 1e-6)

    centers = np.tile(global_med, (n_classes, 1))
    scales = np.tile(global_scale, (n_classes, 1))

    for c in range(n_classes):
        members = train_shape[train_pred == c]
        if members.shape[0] >= 4:
            center = np.median(members, axis=0)
            mad = np.median(np.abs(members - center), axis=0)
            centers[c] = center
            scales[c] = np.maximum(1.4826 * mad, 0.15 * global_scale)

    z = (eval_shape - centers[eval_pred]) / scales[eval_pred]
    return np.sqrt(np.sum(z * z, axis=1))

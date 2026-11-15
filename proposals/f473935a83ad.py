import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "In-scope predictions have intent-specific distributions of centered logit mass across all "
    "classes. Class-conditional surprise in scale-invariant logit skewness and kurtosis detects "
    "atypical winner-versus-field concentration beyond energy or maximum probability."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0164", "m_0008"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape_features(logits):
        x = np.asarray(logits, dtype=float)
        centered = x - np.mean(x, axis=1, keepdims=True)
        scale = np.sqrt(np.mean(centered * centered, axis=1, keepdims=True))
        z = centered / np.maximum(scale, 1e-8)
        skew = np.mean(z ** 3, axis=1)
        kurt = np.mean(z ** 4, axis=1)
        return np.column_stack((skew, kurt))

    train_feat = shape_features(tr.logits)
    eval_feat = shape_features(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_mean = np.mean(train_feat, axis=0)
    global_var = np.var(train_feat, axis=0) + 1e-6
    n_classes = tr.logits.shape[1]
    means = np.tile(global_mean, (n_classes, 1))
    variances = np.tile(global_var, (n_classes, 1))

    for c in range(n_classes):
        mask = train_pred == c
        count = int(np.sum(mask))
        if count >= 4:
            local = train_feat[mask]
            local_mean = np.mean(local, axis=0)
            local_var = np.var(local, axis=0)
            weight = count / (count + 12.0)
            means[c] = weight * local_mean + (1.0 - weight) * global_mean
            variances[c] = (
                weight * local_var
                + (1.0 - weight) * global_var
                + 1e-6
            )

    diff = eval_feat - means[eval_pred]
    scores = np.sum((diff * diff) / variances[eval_pred], axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)

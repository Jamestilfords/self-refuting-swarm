import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope examples typically have class-specific logit concentration and top-two separation patterns, whereas OOS inputs often produce atypical confidence shapes even when their maximum logit is high. A class-conditional typicality score combines deviations in centered logit norm and top-two margin."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]

    def shape_features(logits):
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        norm = np.sqrt(np.mean(centered * centered, axis=1))
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        return np.column_stack((norm, margin))

    train_shape = shape_features(tr.logits)
    eval_shape = shape_features(ev.logits)

    global_mean = np.mean(train_shape, axis=0)
    global_scale = np.std(train_shape, axis=0)
    global_scale = np.maximum(global_scale, 1e-6)

    means = np.tile(global_mean, (k, 1))
    scales = np.tile(global_scale, (k, 1))
    for c in range(k):
        members = train_shape[tr.labels == c]
        if members.shape[0] >= 2:
            means[c] = np.mean(members, axis=0)
            scales[c] = np.maximum(np.std(members, axis=0), 1e-6)

    predicted = np.argmax(ev.logits, axis=1)
    z = (eval_shape - means[predicted]) / scales[predicted]
    return np.sum(z * z, axis=1)

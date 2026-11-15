import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The entropy of normalized Dirichlet concentrations measures how evidence is allocated across intents, independently of total evidence. "
    "Class-conditional deviations from the in-scope allocation sharpness flag evidential shapes atypical for the intent receiving the most support."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0041"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape_entropy(alpha):
        a = np.maximum(np.asarray(alpha, dtype=float), 1e-12)
        p = a / np.sum(a, axis=1, keepdims=True)
        return -np.sum(p * np.log(p), axis=1) / np.log(float(p.shape[1]))

    train_alpha = ctx.h.alpha(tr.logits)
    eval_alpha = ctx.h.alpha(ev.logits)
    train_shape = shape_entropy(train_alpha)
    eval_shape = shape_entropy(eval_alpha)

    train_pred = np.argmax(train_alpha, axis=1)
    eval_pred = np.argmax(eval_alpha, axis=1)
    k = train_alpha.shape[1]

    global_med = np.median(train_shape)
    global_mad = np.median(np.abs(train_shape - global_med)) + 1e-6

    med = np.full(k, global_med, dtype=float)
    mad = np.full(k, global_mad, dtype=float)

    for c in range(k):
        values = train_shape[train_pred == c]
        if values.size >= 8:
            center = np.median(values)
            scale = np.median(np.abs(values - center)) + 1e-6
            med[c] = center
            mad[c] = scale

    scores = np.abs(eval_shape - med[eval_pred]) / mad[eval_pred]
    return np.maximum(scores, 0.0)

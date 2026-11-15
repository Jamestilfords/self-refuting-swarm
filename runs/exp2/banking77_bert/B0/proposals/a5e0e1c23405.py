import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "EDL OOS inputs can exhibit an unusual Dirichlet shape, combining abnormal total evidence "
    "with atypical concentration dominance for the classifier-assigned intent. Class-conditional "
    "train-fitted robust deviations capture this evidence-shape surprise beyond vacuity alone."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    a_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), eps)

    def shape_features(alpha):
        strength = np.log(np.sum(alpha, axis=1))
        probs = alpha / np.sum(alpha, axis=1, keepdims=True)
        dominance = np.log(np.maximum(np.max(probs, axis=1), eps))
        entropy = -np.sum(probs * np.log(np.maximum(probs, eps)), axis=1)
        return np.column_stack((strength, dominance, entropy))

    x_tr = shape_features(a_tr)
    x_ev = shape_features(a_ev)
    pred_ev = np.argmax(ev.logits, axis=1)
    labels = tr.labels.astype(int)
    k = tr.logits.shape[1]

    global_med = np.median(x_tr, axis=0)
    global_mad = np.median(np.abs(x_tr - global_med), axis=0) * 1.4826 + eps

    med = np.tile(global_med, (k, 1))
    scale = np.tile(global_mad, (k, 1))
    for c in range(k):
        xc = x_tr[labels == c]
        if xc.shape[0] >= 4:
            mc = np.median(xc, axis=0)
            sc = np.median(np.abs(xc - mc), axis=0) * 1.4826
            med[c] = mc
            scale[c] = np.maximum(sc, 0.05 * global_mad + eps)

    z = np.abs((x_ev - med[pred_ev]) / scale[pred_ev])
    return z[:, 0] + z[:, 1] + 0.5 * z[:, 2]

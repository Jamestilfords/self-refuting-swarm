import numpy as np

FAMILY = "logit"
ARM = "edl"
RATIONALE = (
    "The collision entropy of normalized Dirichlet concentrations measures how diffusely "
    "evidence is allocated across intents, independent of total evidential mass. OOS queries "
    "should have unusually diffuse allocation relative to training queries predicted as the same intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0041"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8
    k = tr.logits.shape[1]

    def diffuseness(logits):
        alpha = np.asarray(ctx.h.alpha(logits), dtype=float)
        alpha = np.nan_to_num(alpha, nan=1.0, posinf=1e12, neginf=eps)
        alpha = np.clip(alpha, eps, 1e12)
        probs = alpha / np.maximum(alpha.sum(axis=1, keepdims=True), eps)
        return 1.0 - np.sum(probs * probs, axis=1)

    train_shape = diffuseness(tr.logits)
    eval_shape = diffuseness(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    counts = np.bincount(train_pred, minlength=k).astype(float)
    sums = np.bincount(train_pred, weights=train_shape, minlength=k)
    global_mean = float(np.mean(train_shape))
    means = np.where(counts > 0, sums / np.maximum(counts, 1.0), global_mean)

    centered_sq = (train_shape - means[train_pred]) ** 2
    var_sums = np.bincount(train_pred, weights=centered_sq, minlength=k)
    global_scale = float(np.std(train_shape))
    scales = np.sqrt(var_sums / np.maximum(counts, 1.0))
    scales = np.where(counts > 1, scales, global_scale)
    scales = np.maximum(scales, 1e-6)

    return (eval_shape - means[eval_pred]) / scales[eval_pred]

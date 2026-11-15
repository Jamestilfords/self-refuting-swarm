import numpy as np

FAMILY = "knn"
ARM = "edl"
RATIONALE = "OOS inputs can have atypical Dirichlet concentration profiles even when their total evidence resembles in-scope examples. A kNN score over centered log-concentrations plus total concentration measures unusual evidence shape rather than a monotone transform of energy."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def evidence_shape(logits):
        alpha = np.asarray(ctx.h.alpha(logits), dtype=float)
        alpha = np.clip(alpha, 1e-8, 1e12)
        log_alpha = np.log(alpha)
        centered = log_alpha - log_alpha.mean(axis=1, keepdims=True)
        strength = np.log(alpha.sum(axis=1, keepdims=True))
        return np.concatenate([centered, strength], axis=1)

    train_x = evidence_shape(tr.logits)
    eval_x = evidence_shape(ev.logits)

    med = np.median(train_x, axis=0, keepdims=True)
    mad = np.median(np.abs(train_x - med), axis=0, keepdims=True)
    scale = np.maximum(1.4826 * mad, 1e-6)

    train_z = (train_x - med) / scale
    eval_z = (eval_x - med) / scale
    return ctx.h.knn(train_z, eval_z, k=15, l2=True)

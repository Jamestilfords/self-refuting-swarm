import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "OOS inputs can produce weak, diffuse Dirichlet evidence spread across many intents, yielding a high concentration-shape entropy even when total concentration alone is not decisive. Predicted-intent empirical calibration accounts for intents whose in-scope evidence distributions are naturally broader."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape_entropy(alpha):
        a = np.maximum(np.asarray(alpha, dtype=float), 1e-12)
        p = a / np.sum(a, axis=1, keepdims=True)
        k = p.shape[1]
        return -np.sum(p * np.log(p), axis=1) / np.log(float(max(k, 2)))

    train_alpha = ctx.h.alpha(tr.logits)
    eval_alpha = ctx.h.alpha(ev.logits)

    train_shape = shape_entropy(train_alpha)
    eval_shape = shape_entropy(eval_alpha)

    train_pred = np.argmax(train_alpha, axis=1)
    eval_pred = np.argmax(eval_alpha, axis=1)

    global_sorted = np.sort(train_shape)
    out = np.empty(eval_shape.shape[0], dtype=float)

    for c in range(train_alpha.shape[1]):
        mask_train = train_pred == c
        mask_eval = eval_pred == c
        reference = np.sort(train_shape[mask_train])
        if reference.size < 8:
            reference = global_sorted
        if np.any(mask_eval):
            out[mask_eval] = (
                np.searchsorted(reference, eval_shape[mask_eval], side="right")
                / float(reference.size + 1)
            )

    return np.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)

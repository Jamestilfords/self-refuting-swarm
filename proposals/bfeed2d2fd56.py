import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "In-scope examples exhibit characteristic confidence responses as logits are softened at "
    "several temperatures. A class-conditional deviation score on this tempered entropy curve "
    "flags atypical logit scale and competition patterns beyond a single-temperature confidence."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    temperatures = np.array([0.5, 1.0, 2.0, 4.0], dtype=float)

    def tempered_entropy(logits):
        x = logits[:, None, :] / temperatures[None, :, None]
        x = x - np.max(x, axis=2, keepdims=True)
        p = np.exp(x)
        p /= np.sum(p, axis=2, keepdims=True)
        return -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=2)

    train_curve = tempered_entropy(np.asarray(tr.logits, dtype=float))
    eval_curve = tempered_entropy(np.asarray(ev.logits, dtype=float))
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_mean = np.mean(train_curve, axis=0)
    global_var = np.var(train_curve, axis=0) + 1e-6
    means = np.tile(global_mean, (n_classes, 1))
    variances = np.tile(global_var, (n_classes, 1))

    for c in range(n_classes):
        mask = train_pred == c
        if np.sum(mask) >= 8:
            curve_c = train_curve[mask]
            means[c] = np.mean(curve_c, axis=0)
            variances[c] = np.var(curve_c, axis=0) + 1e-6

    z = (eval_curve - means[eval_pred]) / np.sqrt(variances[eval_pred])
    return np.mean(z * z, axis=1)

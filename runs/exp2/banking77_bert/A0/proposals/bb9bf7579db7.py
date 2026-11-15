import numpy as np

FAMILY = "logit"
ARM = "edl"
RATIONALE = (
    "OOS inputs should distribute Dirichlet concentration more evenly across intent classes, "
    "whereas in-scope inputs typically exhibit a peaked concentration shape. Entropy of normalized "
    "Dirichlet concentrations measures this allocation pattern independently of total evidence."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def concentration_entropy(logits):
        alpha = np.maximum(ctx.h.alpha(logits), 1e-12)
        p = alpha / np.sum(alpha, axis=1, keepdims=True)
        ent = -np.sum(p * np.log(p), axis=1)
        return ent / np.log(float(alpha.shape[1]))

    train_entropy = concentration_entropy(tr.logits)
    eval_entropy = concentration_entropy(ev.logits)

    center = np.median(train_entropy)
    scale = np.median(np.abs(train_entropy - center))
    scale = max(float(scale), 1e-6)

    z = (eval_entropy - center) / (1.4826 * scale)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))

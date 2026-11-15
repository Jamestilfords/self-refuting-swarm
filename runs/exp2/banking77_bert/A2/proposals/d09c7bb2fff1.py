import numpy as np

FAMILY = "knn"
ARM = "edl"
RATIONALE = (
    "The centered log-ratio transform isolates the relative allocation pattern of "
    "Dirichlet concentrations while removing total-evidence magnitude. OOS queries "
    "should have less local support among in-scope evidential shape patterns."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0017"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def evidential_shape(logits):
        alpha = np.asarray(ctx.h.alpha(logits), dtype=float)
        alpha = np.clip(alpha, 1e-12, 1e12)
        log_alpha = np.log(alpha)
        clr = log_alpha - np.mean(log_alpha, axis=1, keepdims=True)
        norm = np.linalg.norm(clr, axis=1, keepdims=True)
        return clr / np.maximum(norm, 1e-12)

    train_shape = evidential_shape(tr.logits)
    eval_shape = evidential_shape(ev.logits)
    return ctx.h.knn(train_shape, eval_shape, k=20, l2=True)

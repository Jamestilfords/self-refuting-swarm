import numpy as np

FAMILY = "knn"
ARM = "edl"
RATIONALE = "The normalized Dirichlet concentration vector captures evidential shape independently of total evidence, so atypical class-relative evidence patterns can be detected even when energy is similar. Nearest-neighbor distance in centered log-ratio concentration space measures departure from training-time evidential geometry."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    def clr_alpha(logits):
        alpha = np.asarray(ctx.h.alpha(logits), dtype=float)
        alpha = np.clip(alpha, 1e-12, 1e12)
        comp = alpha / np.sum(alpha, axis=1, keepdims=True)
        log_comp = np.log(np.clip(comp, 1e-12, 1.0))
        return log_comp - np.mean(log_comp, axis=1, keepdims=True)

    train_shape = clr_alpha(ctx.train.logits)
    eval_shape = clr_alpha(ctx.eval.logits)
    return ctx.h.knn(train_shape, eval_shape, k=15, l2=True)

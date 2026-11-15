import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The centered log-ratio of Dirichlet concentrations captures relative evidence allocation independently "
    "of overall evidence magnitude. A class-conditional, variance-normalized deviation from the learned "
    "evidential competition template flags OOS inputs with atypical intent-evidence structure."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0050", "m_0025"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    alpha_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    alpha_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    clr_tr = np.log(alpha_tr)
    clr_tr = clr_tr - clr_tr.mean(axis=1, keepdims=True)
    clr_ev = np.log(alpha_ev)
    clr_ev = clr_ev - clr_ev.mean(axis=1, keepdims=True)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_mean = clr_tr.mean(axis=0)
    global_var = clr_tr.var(axis=0) + 1e-5

    counts = np.bincount(pred_tr, minlength=n_classes).astype(float)
    sums = np.zeros((n_classes, clr_tr.shape[1]), dtype=float)
    np.add.at(sums, pred_tr, clr_tr)

    shrink = 12.0
    class_means = (sums + shrink * global_mean) / (counts[:, None] + shrink)

    residual_tr = clr_tr - class_means[pred_tr]
    class_var_sums = np.zeros_like(class_means)
    np.add.at(class_var_sums, pred_tr, residual_tr * residual_tr)
    class_vars = (class_var_sums + shrink * global_var) / (counts[:, None] + shrink)
    class_vars = np.maximum(class_vars, 1e-5)

    residual_ev = clr_ev - class_means[pred_ev]
    scores = np.mean((residual_ev * residual_ev) / class_vars[pred_ev], axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)

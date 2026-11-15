import numpy as np
from scipy.special import gammaln, digamma

FAMILY = "other"
ARM = "edl"
RATIONALE = "Dirichlet differential entropy and concentration imbalance capture the shape of evidential concentrations, not merely their total evidence or energy. OOS inputs should exhibit evidence-shape profiles atypical for the predicted in-scope class."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    def shape(alpha):
        total = np.sum(alpha, axis=1)
        p = alpha / total[:, None]
        k = alpha.shape[1]
        # Negative Dirichlet entropy is sensitive to both total concentration
        # and how evidence is distributed among classes.
        log_b = np.sum(gammaln(alpha), axis=1) - gammaln(total)
        d_entropy = (
            log_b
            + (total - k) * digamma(total)
            - np.sum((alpha - 1.0) * digamma(alpha), axis=1)
        )
        neg_entropy = -d_entropy
        concentration = np.sum(p * p, axis=1)
        top_gap = np.max(p, axis=1) - np.partition(p, -2, axis=1)[:, -2]
        return np.column_stack((neg_entropy, concentration, top_gap))

    z_tr = shape(a_tr)
    z_ev = shape(a_ev)
    labels = tr.labels.astype(int)
    pred_ev = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_mu = np.mean(z_tr, axis=0)
    global_var = np.var(z_tr, axis=0) + 1e-6
    means = np.tile(global_mu, (n_classes, 1))
    vars_ = np.tile(global_var, (n_classes, 1))

    for c in range(n_classes):
        part = z_tr[labels == c]
        if part.shape[0] >= 3:
            means[c] = np.mean(part, axis=0)
            vars_[c] = np.var(part, axis=0) + 1e-6

    diff = z_ev - means[pred_ev]
    return np.sum((diff * diff) / vars_[pred_ev], axis=1)

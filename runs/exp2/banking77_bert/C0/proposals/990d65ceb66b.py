import numpy as np
from scipy.special import digamma, gammaln

FAMILY = "other"
ARM = "edl"
RATIONALE = "This measures the symmetric KL divergence between each item's predicted Dirichlet distribution and the closest class-conditional Dirichlet prototype learned on training data. It uses both concentration shape and total evidence, so it is not a monotone transformation of energy or vacuity."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_train = np.clip(ctx.h.alpha(tr.logits), 1e-6, 1e6)
    a_eval = np.clip(ctx.h.alpha(ev.logits), 1e-6, 1e6)
    labels = tr.labels
    classes = np.unique(labels)

    q0 = np.sum(a_eval, axis=1)
    log_bases_q = gammaln(q0) - np.sum(gammaln(a_eval), axis=1)
    psi_q = digamma(a_eval) - digamma(q0)[:, None]

    best = np.full(a_eval.shape[0], np.inf)

    for c in classes:
        proto = np.median(a_train[labels == c], axis=0)
        proto = np.clip(proto, 1e-6, 1e6)
        p0 = np.sum(proto)

        log_base_p = gammaln(p0) - np.sum(gammaln(proto))
        psi_p = digamma(proto) - digamma(p0)

        kl_qp = (
            log_bases_q
            - log_base_p
            + np.sum((a_eval - proto) * psi_q, axis=1)
        )
        kl_pq = (
            log_base_p
            - log_bases_q
            + np.sum((proto - a_eval) * psi_p, axis=1)
        )

        divergence = 0.5 * (kl_qp + kl_pq)
        best = np.minimum(best, divergence)

    return np.nan_to_num(best, nan=1e12, posinf=1e12, neginf=0.0)

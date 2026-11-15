import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "A ViM-style virtual OOS logit is formed from the fraction of feature energy outside the principal in-distribution subspace. Comparing this virtual logit directly against the classifier logit partition yields a ranking distinct from an additive residual-energy score."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    pca = PCA(n_components=64, svd_solver="randomized", random_state=2718)
    pca.fit(tr.pooled)

    def residual_fraction(x):
        centered = x - pca.mean_
        total = np.sum(centered * centered, axis=1)
        kept = pca.transform(x)
        kept = np.sum(kept * kept, axis=1)
        residual = np.maximum(total - kept, 0.0)
        return np.sqrt(residual / (total + 1e-8))

    train_res = residual_fraction(tr.pooled)
    eval_res = residual_fraction(ev.pooled)

    train_peak = np.max(tr.logits, axis=1)
    alpha = np.mean(np.maximum(train_peak, 0.0)) / (np.mean(train_res) + 1e-8)
    alpha = float(np.clip(alpha, 1e-4, 1e4))

    virtual_logit = alpha * eval_res
    m = np.maximum(np.max(ev.logits, axis=1), virtual_logit)
    log_known = m + np.log(np.sum(np.exp(ev.logits - m[:, None]), axis=1))
    return virtual_logit - np.logaddexp(log_known, virtual_logit)

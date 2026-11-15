import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "The residual norm outside a global in-scope principal subspace measures encoder variation "
    "unsupported by training representations, while energy captures classifier uncertainty. "
    "Train-calibrated tail scores combine geometric novelty with uncertain intent assignment."
)
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0069", "m_0059"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(128, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=2718)
    z_train = pca.fit_transform(tr.pooled)
    z_eval = pca.transform(ev.pooled)

    train_resid = np.sum((tr.pooled - pca.inverse_transform(z_train)) ** 2, axis=1)
    eval_resid = np.sum((ev.pooled - pca.inverse_transform(z_eval)) ** 2, axis=1)

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    def upper_tail(train_values, values):
        ref = np.sort(np.nan_to_num(train_values, nan=0.0, posinf=1e30, neginf=-1e30))
        val = np.nan_to_num(values, nan=0.0, posinf=1e30, neginf=-1e30)
        return np.searchsorted(ref, val, side="right") / float(ref.size)

    residual_tail = upper_tail(train_resid, eval_resid)
    energy_tail = upper_tail(train_energy, eval_energy)

    return 0.65 * residual_tail + 0.35 * energy_tail + 1e-8 * np.nan_to_num(
        eval_resid, nan=0.0, posinf=1e30, neginf=-1e30
    )

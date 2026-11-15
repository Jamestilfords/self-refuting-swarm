import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "A confidence-conditioned switch retains the energy ordering for confident predictions, "
    "but replaces unreliable logit evidence with local representation support for uncertain items."
)
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0114"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    k = 24
    train_knn = np.asarray(ctx.h.knn(tr.pooled, tr.pooled, k=k, l2=True), dtype=float)
    eval_knn = np.asarray(ctx.h.knn(tr.pooled, ev.pooled, k=k, l2=True), dtype=float)

    def empirical_surprise(train_values, values):
        ref = np.sort(np.asarray(train_values, dtype=float))
        ranks = np.searchsorted(ref, np.asarray(values, dtype=float), side="right")
        return ranks.astype(float) / (ref.size + 1.0)

    energy_surprise = empirical_surprise(train_energy, eval_energy)
    knn_surprise = empirical_surprise(train_knn, eval_knn)

    train_msp = np.max(ctx.h.softmax(tr.logits), axis=1)
    eval_msp = np.max(ctx.h.softmax(ev.logits), axis=1)

    center = float(np.quantile(train_msp, 0.35))
    q25, q75 = np.quantile(train_msp, [0.25, 0.75])
    scale = max(float(q75 - q25), 1e-6)

    uncertainty_gate = 1.0 / (1.0 + np.exp((eval_msp - center) / scale))
    scores = (1.0 - uncertainty_gate) * energy_surprise + uncertainty_gate * knn_surprise
    return np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)

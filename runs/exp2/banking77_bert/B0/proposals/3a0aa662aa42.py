import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy-based confidence and encoder-space Gaussian support capture complementary forms "
    "of in-distribution evidence. Train-fitted robust calibrators convert both into ID-support "
    "probabilities, whose joint negative log-support is high when either signal rejects a query."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0071"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = ctx.h.energy(tr.logits)
    train_log_density = g.log_density(tr.pooled)

    energy_id = ctx.h.normalizer("qsigmoid", -train_energy)
    density_id = ctx.h.normalizer("qsigmoid", train_log_density)

    p_energy = np.clip(energy_id(-ctx.h.energy(ev.logits)), 1e-12, 1.0)
    p_density = np.clip(density_id(g.log_density(ev.pooled)), 1e-12, 1.0)

    return -np.log(p_energy) - np.log(p_density)

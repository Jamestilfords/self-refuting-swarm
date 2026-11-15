import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This calibrated switch preserves energy ordering for low-entropy, confident predictions, but transitions toward pooled-feature density novelty for high-entropy predictions. The nonlinear uncertainty gate makes ambiguous items depend primarily on whether they occupy low-density encoder regions."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    density_normalizer = ctx.h.normalizer("ecdf", g.log_density(tr.pooled))
    density_oos = 1.0 - density_normalizer(g.log_density(ev.pooled))

    sorted_energy = np.sort(train_energy)
    energy_oos = np.searchsorted(
        sorted_energy, eval_energy, side="right"
    ).astype(float) / float(len(sorted_energy))

    train_entropy = ctx.h.entropy(tr.logits)
    eval_entropy = ctx.h.entropy(ev.logits)
    center = float(np.percentile(train_entropy, 85))
    scale = float(
        np.percentile(train_entropy, 75) -
        np.percentile(train_entropy, 25)
    )
    scale = max(scale, 1e-8)

    z = np.clip((eval_entropy - center) / scale, -30.0, 30.0)
    uncertain = 1.0 / (1.0 + np.exp(-z))

    return (1.0 - uncertain) * energy_oos + uncertain * density_oos

import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This preserves the energy ranking for low-entropy, confident predictions, while progressively adding pooled-feature density evidence when classifier uncertainty is high. Feature atypicality can therefore rescue uncertain examples whose logits alone are ambiguous."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_ld = g.log_density(tr.pooled)
    density_normalizer = ctx.h.normalizer("ecdf", train_ld)
    feature_oos = 1.0 - density_normalizer(g.log_density(ev.pooled))

    train_entropy = ctx.h.entropy(tr.logits)
    eval_entropy = ctx.h.entropy(ev.logits)
    lo, hi = np.quantile(train_entropy, [0.50, 0.90])
    spread = max(float(hi - lo), 1e-6)
    uncertainty_gate = np.clip((eval_entropy - lo) / spread, 0.0, 1.0)

    train_energy = ctx.h.energy(tr.logits)
    q25, q75 = np.quantile(train_energy, [0.25, 0.75])
    energy_scale = max(float(q75 - q25), 1e-6)

    scores = ctx.h.energy(ev.logits) + energy_scale * uncertainty_gate * feature_oos
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)

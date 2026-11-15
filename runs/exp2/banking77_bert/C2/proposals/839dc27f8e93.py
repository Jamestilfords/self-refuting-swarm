import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This retains the energy ranking for low-entropy, confident predictions, while progressively adding pooled-feature Mahalanobis evidence when the classifier's logits are uncertain. Feature distance can distinguish boundary uncertainty from queries that are genuinely outside the learned intent manifold."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_entropy = np.asarray(ctx.h.entropy(tr.logits), dtype=float)
    eval_entropy = np.asarray(ctx.h.entropy(ev.logits), dtype=float)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e25, e75 = np.quantile(train_energy, [0.25, 0.75])
    energy_scale = max(float(e75 - e25), 1e-6)

    h50, h90 = np.quantile(train_entropy, [0.50, 0.90])
    entropy_scale = max(float(h90 - h50), 1e-6)
    gate = 1.0 / (1.0 + np.exp(-np.clip(
        (eval_entropy - h50) / entropy_scale, -30.0, 30.0
    )))

    d50, d90 = np.quantile(train_dist, [0.50, 0.90])
    dist_scale = max(float(d90 - d50), 1e-6)
    dist_evidence = np.clip((eval_dist - d50) / dist_scale, -5.0, 10.0)

    scores = eval_energy + energy_scale * gate * dist_evidence
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)

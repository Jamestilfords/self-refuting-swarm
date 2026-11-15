FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Train-calibrated empirical percentiles put classifier energy and nearest-class encoder Mahalanobis distance on a common scale. Their concordant upper-tail interaction emphasizes queries that are simultaneously classifier-uncertain and far from every intent manifold."
SOURCE = "https://www.emergentmind.com/papers/2101.03778"
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0001", "m_0003"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_distance = g.min_mahalanobis_sq(tr.pooled)
    eval_distance = g.min_mahalanobis_sq(ev.pooled)

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    distance_cdf = ctx.h.normalizer("ecdf", train_distance)
    energy_cdf = ctx.h.normalizer("ecdf", train_energy)

    d = distance_cdf(eval_distance)
    e = energy_cdf(eval_energy)

    return 0.5 * (d + e) + 0.5 * d * e

import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Predictive entropy captures unfamiliar or ambiguous intent competition, while Mahalanobis distance on L2-normalized bottleneck features measures whether the representation is supported by an in-scope intent. Train-fitted upper-tail probabilities put these complementary novelty signals on a common scale before Fisher fusion."
SOURCE = "https://arxiv.org/html/2403.10803"
PARENTS = []
MEMORY_USED = ["m_0186", "m_0081"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_entropy = ctx.h.entropy(tr.logits)
    eval_entropy = ctx.h.entropy(ev.logits)

    train_feats = ctx.h.l2(tr.feats)
    eval_feats = ctx.h.l2(ev.feats)
    gda = ctx.h.gda(train_feats, tr.labels, cov="tied", whiten=True)
    train_distance = gda.min_mahalanobis_sq(train_feats)
    eval_distance = gda.min_mahalanobis_sq(eval_feats)

    def upper_tail(train_values, values):
        reference = np.sort(np.asarray(train_values, dtype=float))
        positions = np.searchsorted(reference, values, side="left")
        counts_ge = reference.size - positions
        return (counts_ge + 1.0) / (reference.size + 1.0)

    p_entropy = upper_tail(train_entropy, eval_entropy)
    p_distance = upper_tail(train_distance, eval_distance)

    return -2.0 * (np.log(p_entropy) + np.log(p_distance))

import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Maximum-softmax uncertainty and bottleneck-space nearest-neighbor distance capture complementary forms of OOS behavior. Training-fitted quantile sigmoids place both anomaly signals on a common bounded scale before averaging."
SOURCE = ""
PARENTS = ["msp", "d_port"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_msp = ctx.h.msp(tr.logits)
    eval_msp = ctx.h.msp(ev.logits)

    train_knn = ctx.h.knn(tr.feats, tr.feats, k=11, l2=True)
    eval_knn = ctx.h.knn(tr.feats, ev.feats, k=10, l2=True)

    msp_normalizer = ctx.h.normalizer("qsigmoid", train_msp)
    knn_normalizer = ctx.h.normalizer("qsigmoid", train_knn)

    msp_score = msp_normalizer(eval_msp)
    knn_score = knn_normalizer(eval_knn)

    return 0.45 * msp_score + 0.55 * knn_score

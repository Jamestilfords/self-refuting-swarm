import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Bottleneck density can capture near-OOS examples that resemble known encoder domains but miss "
    "intent structure, while pooled encoder density detects broader domain shifts. Training-derived "
    "inverse-variance weights place both negative log-density signals on a comparable additive scale."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_bottleneck = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_encoder = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_b = g_bottleneck.neg_log_density(tr.feats)
    train_e = g_encoder.neg_log_density(tr.pooled)
    eval_b = g_bottleneck.neg_log_density(ev.feats)
    eval_e = g_encoder.neg_log_density(ev.pooled)

    med_b = np.median(train_b)
    med_e = np.median(train_e)
    scale_b = np.sqrt(np.mean((train_b - med_b) ** 2)) + 1e-8
    scale_e = np.sqrt(np.mean((train_e - med_e) ** 2)) + 1e-8

    weight_b = 1.0 / scale_b
    weight_e = 1.0 / scale_e
    weight_sum = weight_b + weight_e

    scores = (weight_b * (eval_b - med_b) + weight_e * (eval_e - med_e)) / weight_sum
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)

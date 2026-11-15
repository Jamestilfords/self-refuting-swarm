import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Mahalanobis density in bottleneck space emphasizes fine-grained, near-intent deviations, while encoder-space density captures broader domain shifts. Training-set robust scales calibrate their additive weights without using evaluation statistics."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_port = g_port.min_mahalanobis_sq(tr.feats)
    train_full = g_full.min_mahalanobis_sq(tr.pooled)
    eval_port = g_port.min_mahalanobis_sq(ev.feats)
    eval_full = g_full.min_mahalanobis_sq(ev.pooled)

    med_port = np.median(train_port)
    med_full = np.median(train_full)
    scale_port = np.quantile(train_port, 0.75) - np.quantile(train_port, 0.25)
    scale_full = np.quantile(train_full, 0.75) - np.quantile(train_full, 0.25)

    scale_port = max(float(scale_port), 1e-8)
    scale_full = max(float(scale_full), 1e-8)

    w_port = 1.0 / scale_port
    w_full = 1.0 / scale_full
    w_sum = w_port + w_full
    w_port /= w_sum
    w_full /= w_sum

    scores = w_port * (eval_port - med_port) / scale_port
    scores += w_full * (eval_full - med_full) / scale_full
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)

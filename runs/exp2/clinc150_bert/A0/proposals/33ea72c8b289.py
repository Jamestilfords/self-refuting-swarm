import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Variance-normalized additive fusion of bottleneck and encoder Mahalanobis distances combines sensitivity to near-OOS held-out intents with sensitivity to far-domain OOS queries."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_port = g_port.min_mahalanobis_sq(tr.feats)
    train_full = g_full.min_mahalanobis_sq(tr.pooled)
    eval_port = g_port.min_mahalanobis_sq(ev.feats)
    eval_full = g_full.min_mahalanobis_sq(ev.pooled)

    mu_port = np.mean(train_port)
    mu_full = np.mean(train_full)
    sd_port = max(float(np.std(train_port)), 1e-8)
    sd_full = max(float(np.std(train_full)), 1e-8)

    w_port = 1.0 / sd_port
    w_full = 1.0 / sd_full
    total = w_port + w_full

    scores = (
        w_port * (eval_port - mu_port) / sd_port
        + w_full * (eval_full - mu_full) / sd_full
    ) / total
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)

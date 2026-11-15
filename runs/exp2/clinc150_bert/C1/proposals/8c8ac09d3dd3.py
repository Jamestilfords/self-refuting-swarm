import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck and encoder-space class-conditional densities capture complementary near-OOS and far-OOS deviations. Train-fitted inverse-scale weights make their negative log-density contributions comparable without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_port = np.asarray(g_port.neg_log_density(tr.feats), dtype=float)
    train_full = np.asarray(g_full.neg_log_density(tr.pooled), dtype=float)
    eval_port = np.asarray(g_port.neg_log_density(ev.feats), dtype=float)
    eval_full = np.asarray(g_full.neg_log_density(ev.pooled), dtype=float)

    med_port = np.median(train_port)
    med_full = np.median(train_full)
    scale_port = np.median(np.abs(train_port - med_port)) * 1.4826
    scale_full = np.median(np.abs(train_full - med_full)) * 1.4826

    scale_port = max(float(scale_port), 1e-8)
    scale_full = max(float(scale_full), 1e-8)

    w_port = 1.0 / scale_port
    w_full = 1.0 / scale_full
    weight_sum = w_port + w_full
    w_port /= weight_sum
    w_full /= weight_sum

    scores = w_port * ((eval_port - med_port) / scale_port)
    scores += w_full * ((eval_full - med_full) / scale_full)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)

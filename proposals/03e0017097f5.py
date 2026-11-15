FAMILY = "fusion"
ARM = "edl"
RATIONALE = "ER-1 grid: GDA log-density on bottleneck features (PCA 128), coupled to the logits by minmax x mul, EDL arm."
SOURCE = ""
PARENTS = ["energy", "d_port"]
MEMORY_USED = []


def score(ctx):
    tr, ev, h = ctx.train, ctx.eval, ctx.h
    x_tr, x_ev = tr.feats, ev.feats
    g = h.gda(x_tr, tr.labels, pca=128, whiten=True)
    d_ev = g.neg_log_density(x_ev)
    norm = h.normalizer("minmax", -g.neg_log_density(x_tr))
    p_hat = norm(-d_ev)
    return h.couple_mul(ev.logits, p_hat)

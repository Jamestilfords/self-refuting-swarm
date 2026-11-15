import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "PCA-reduced bottleneck and encoder Gaussian densities retain complementary geometry while avoiding a near-identical full-space ranking. Train-calibrated ECDF tail scores make the two density scales comparable before additive fusion."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    port_dim = min(96, tr.feats.shape[1])
    full_dim = min(192, tr.pooled.shape[1])

    port_pca = ctx.h.pca(tr.feats, port_dim)
    full_pca = ctx.h.pca(tr.pooled, full_dim)

    tr_port = np.asarray(port_pca.transform(tr.feats), dtype=float)
    ev_port = np.asarray(port_pca.transform(ev.feats), dtype=float)
    tr_full = np.asarray(full_pca.transform(tr.pooled), dtype=float)
    ev_full = np.asarray(full_pca.transform(ev.pooled), dtype=float)

    g_port = ctx.h.gda(tr_port, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr_full, tr.labels, cov="tied", whiten=True)

    ld_port_train = np.asarray(g_port.log_density(tr_port), dtype=float)
    ld_full_train = np.asarray(g_full.log_density(tr_full), dtype=float)
    ld_port_eval = np.asarray(g_port.log_density(ev_port), dtype=float)
    ld_full_eval = np.asarray(g_full.log_density(ev_full), dtype=float)

    norm_port = ctx.h.normalizer("ecdf", ld_port_train)
    norm_full = ctx.h.normalizer("ecdf", ld_full_train)

    in_port_train = np.asarray(norm_port(ld_port_train), dtype=float)
    in_full_train = np.asarray(norm_full(ld_full_train), dtype=float)

    var_port = max(float(np.var(in_port_train)), 1e-8)
    var_full = max(float(np.var(in_full_train)), 1e-8)
    w_port = 1.0 / var_port
    w_full = 1.0 / var_full
    weight_sum = w_port + w_full
    w_port /= weight_sum
    w_full /= weight_sum

    port_tail = 1.0 - np.asarray(norm_port(ld_port_eval), dtype=float)
    full_tail = 1.0 - np.asarray(norm_full(ld_full_eval), dtype=float)

    scores = w_port * port_tail + w_full * full_tail
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

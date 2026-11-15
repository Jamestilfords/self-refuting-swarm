import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "The score preserves energy for training-calibrated confident predictions, but activates a class-conditional pooled-feature distance alarm when classifier confidence is unusually low for its predicted intent. This targets forced low-confidence assignments whose representations are also atypical for the assigned class."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0097", "m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    d_tr = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    d_ev = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_tr = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    e_ev = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    p_tr = np.asarray(ctx.h.softmax(tr.logits), dtype=float)
    p_ev = np.asarray(ctx.h.softmax(ev.logits), dtype=float)
    c_tr = np.argmax(p_tr, axis=1)
    c_ev = np.argmax(p_ev, axis=1)
    conf_tr = np.max(p_tr, axis=1)
    conf_ev = np.max(p_ev, axis=1)

    k = tr.logits.shape[1]
    global_q = float(np.quantile(conf_tr, 0.25))
    global_mu = float(np.mean(d_tr))
    global_sd = float(np.std(d_tr))
    if not np.isfinite(global_sd) or global_sd < 1e-8:
        global_sd = 1.0

    q_conf = np.full(k, global_q, dtype=float)
    mu_d = np.full(k, global_mu, dtype=float)
    sd_d = np.full(k, global_sd, dtype=float)

    for cls in range(k):
        mask = c_tr == cls
        if np.sum(mask) >= 8:
            q_conf[cls] = float(np.quantile(conf_tr[mask], 0.25))
            mu_d[cls] = float(np.mean(d_tr[mask]))
            s = float(np.std(d_tr[mask]))
            if np.isfinite(s) and s >= 1e-8:
                sd_d[cls] = s

    threshold = q_conf[c_ev]
    scale = np.maximum(0.03, 0.15 * np.maximum(threshold, 0.05))
    gate = 1.0 / (1.0 + np.exp(np.clip((conf_ev - threshold) / scale, -40.0, 40.0)))

    z_d = (d_ev - mu_d[c_ev]) / sd_d[c_ev]
    feature_alarm = np.logaddexp(0.0, np.clip(z_d, -30.0, 30.0))

    out = e_ev + gate * feature_alarm
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)

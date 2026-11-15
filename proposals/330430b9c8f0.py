import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "The expected feature radius for a classifier-selected intent depends on classifier confidence. "
    "A query is suspicious when its predicted-class Mahalanobis distance is much larger than the train-fitted distance expected at its logit margin."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0032", "m_0060"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    d_tr_all = g.class_mahalanobis_sq(tr.pooled)
    d_ev_all = g.class_mahalanobis_sq(ev.pooled)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    def margin(logits):
        if logits.shape[1] < 2:
            return logits[:, 0]
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    m_tr = margin(tr.logits)
    m_ev = margin(ev.logits)
    d_tr = d_tr_all[np.arange(len(tr.logits)), pred_tr]
    d_ev = d_ev_all[np.arange(len(ev.logits)), pred_ev]

    x_mean = float(np.mean(m_tr))
    x_scale = float(np.std(m_tr))
    x_scale = max(x_scale, 1e-6)
    x_tr = (m_tr - x_mean) / x_scale
    x_ev = (m_ev - x_mean) / x_scale

    global_var = float(np.mean((x_tr - np.mean(x_tr)) ** 2))
    global_cov = float(np.mean((x_tr - np.mean(x_tr)) * (d_tr - np.mean(d_tr))))
    global_beta = global_cov / max(global_var, 1e-6)
    global_intercept = float(np.mean(d_tr) - global_beta * np.mean(x_tr))
    global_resid = d_tr - (global_intercept + global_beta * x_tr)
    global_scale = max(float(np.sqrt(np.mean(global_resid ** 2))), 1e-6)

    out = np.empty(len(ev.logits), dtype=float)
    for c in range(tr.logits.shape[1]):
        idx = pred_tr == c
        test_idx = pred_ev == c
        if not np.any(test_idx):
            continue

        if np.sum(idx) >= 8:
            xc = x_tr[idx]
            yc = d_tr[idx]
            vx = float(np.mean((xc - np.mean(xc)) ** 2))
            beta = float(np.mean((xc - np.mean(xc)) * (yc - np.mean(yc)))) / max(vx, 1e-6)
            intercept = float(np.mean(yc) - beta * np.mean(xc))
            resid = yc - (intercept + beta * xc)
            scale = max(float(np.sqrt(np.mean(resid ** 2))), global_scale * 0.25)
        else:
            beta = global_beta
            intercept = global_intercept
            scale = global_scale

        out[test_idx] = (d_ev[test_idx] - (intercept + beta * x_ev[test_idx])) / scale

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

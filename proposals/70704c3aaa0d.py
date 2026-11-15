import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Bottleneck and pooled-space selected-intent densities capture near-intent and broad domain shifts, respectively. "
    "Class-conditional tail calibration and train-fitted residualization retain pooled novelty not already explained by bottleneck novelty."
)
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8
    k = tr.logits.shape[1]

    gb = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    gp = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    db_tr_all = gb.class_mahalanobis_sq(tr.feats)
    dp_tr_all = gp.class_mahalanobis_sq(tr.pooled)
    db_tr = db_tr_all[np.arange(len(tr.labels)), tr.labels]
    dp_tr = dp_tr_all[np.arange(len(tr.labels)), tr.labels]

    pred_ev = np.argmax(ev.logits, axis=1)
    db_ev_all = gb.class_mahalanobis_sq(ev.feats)
    dp_ev_all = gp.class_mahalanobis_sq(ev.pooled)
    db_ev = db_ev_all[np.arange(len(pred_ev)), pred_ev]
    dp_ev = dp_ev_all[np.arange(len(pred_ev)), pred_ev]

    z_b_tr = np.empty_like(db_tr, dtype=float)
    z_p_tr = np.empty_like(dp_tr, dtype=float)
    z_b_ev = np.empty_like(db_ev, dtype=float)
    z_p_ev = np.empty_like(dp_ev, dtype=float)

    global_mb = np.median(db_tr)
    global_mp = np.median(dp_tr)
    global_sb = np.median(np.abs(db_tr - global_mb)) + eps
    global_sp = np.median(np.abs(dp_tr - global_mp)) + eps

    for c in range(k):
        idx = tr.labels == c
        if np.sum(idx) >= 4:
            mb = np.median(db_tr[idx])
            mp = np.median(dp_tr[idx])
            sb = np.median(np.abs(db_tr[idx] - mb)) + eps
            sp = np.median(np.abs(dp_tr[idx] - mp)) + eps
        else:
            mb, mp, sb, sp = global_mb, global_mp, global_sb, global_sp

        z_b_tr[idx] = (db_tr[idx] - mb) / sb
        z_p_tr[idx] = (dp_tr[idx] - mp) / sp

        eidx = pred_ev == c
        z_b_ev[eidx] = (db_ev[eidx] - mb) / sb
        z_p_ev[eidx] = (dp_ev[eidx] - mp) / sp

    xb = np.maximum(z_b_tr, 0.0)
    xp = np.maximum(z_p_tr, 0.0)
    denom = np.dot(xb, xb) + eps
    beta = max(0.0, float(np.dot(xb, xp) / denom))

    residual_tr = xp - beta * xb
    residual_scale = np.median(np.abs(residual_tr - np.median(residual_tr))) + eps

    xb_ev = np.maximum(z_b_ev, 0.0)
    xp_ev = np.maximum(z_p_ev, 0.0)
    residual_ev = np.maximum(xp_ev - beta * xb_ev, 0.0) / residual_scale

    bottleneck_scale = np.median(np.abs(xb - np.median(xb))) + eps
    return xb_ev / bottleneck_scale + residual_ev

import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The fraction of non-winning evidence assigned to the strongest rival captures "
    "the shape of evidential competition independently of total Dirichlet mass. "
    "A predicted-intent-specific residual after fitting its normal dependence on total evidence flags atypical OOS competition patterns."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0050", "m_0142"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-10

    a_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), eps)
    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    def competition_features(alpha, pred):
        n, k = alpha.shape
        total = np.sum(alpha, axis=1)
        own = alpha[np.arange(n), pred]
        rival_total = np.maximum(total - own, eps)

        masked = alpha.copy()
        masked[np.arange(n), pred] = 0.0
        strongest_rival = np.max(masked, axis=1)

        rival_share = np.clip(
            strongest_rival / rival_total, eps, 1.0 - eps
        )
        shape = np.log(rival_share) - np.log1p(-rival_share)
        mass = np.log(total)
        return mass, shape

    mass_tr, shape_tr = competition_features(a_tr, pred_tr)
    mass_ev, shape_ev = competition_features(a_ev, pred_ev)

    global_x = np.column_stack([np.ones(len(mass_tr)), mass_tr])
    global_beta = np.linalg.solve(
        global_x.T @ global_x + 1e-5 * np.eye(2),
        global_x.T @ shape_tr,
    )
    global_resid = shape_tr - global_x @ global_beta
    global_center = np.median(global_resid)
    global_scale = np.median(np.abs(global_resid - global_center)) * 1.4826 + 1e-6

    k = tr.logits.shape[1]
    beta = np.tile(global_beta, (k, 1))
    center = np.full(k, global_center)
    scale = np.full(k, global_scale)

    for c in range(k):
        idx = pred_tr == c
        if np.sum(idx) >= 8:
            x = np.column_stack([np.ones(np.sum(idx)), mass_tr[idx]])
            y = shape_tr[idx]
            b = np.linalg.solve(
                x.T @ x + 1e-4 * np.eye(2),
                x.T @ y,
            )
            resid = y - x @ b
            med = np.median(resid)
            mad = np.median(np.abs(resid - med)) * 1.4826
            if np.isfinite(mad) and mad > 1e-6:
                beta[c] = b
                center[c] = med
                scale[c] = mad

    expected = beta[pred_ev, 0] + beta[pred_ev, 1] * mass_ev
    residual = shape_ev - expected
    scores = np.abs(residual - center[pred_ev]) / scale[pred_ev]
    return np.maximum(scores, 0.0)

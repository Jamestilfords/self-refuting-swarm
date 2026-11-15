import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "The log-arithmetic versus log-geometric concentration gap measures how unevenly Dirichlet evidence is distributed across intents, separately from total evidence or vacuity. Regressing this shape statistic on total concentration within each predicted intent flags evidence allocations that are atypical for in-scope examples."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape_and_scale(logits):
        alpha = np.maximum(ctx.h.alpha(logits), 1e-8)
        la = np.log(alpha)
        log_total = np.log(np.sum(alpha, axis=1))
        shape = np.log(np.mean(alpha, axis=1)) - np.mean(la, axis=1)
        return shape, log_total, np.argmax(logits, axis=1)

    y, x, pred = shape_and_scale(tr.logits)
    ey, ex, epred = shape_and_scale(ev.logits)

    n_classes = tr.logits.shape[1]
    global_x = np.column_stack([np.ones_like(x), x])
    beta_global = np.linalg.solve(
        global_x.T @ global_x + 1e-6 * np.eye(2),
        global_x.T @ y,
    )
    global_resid = y - global_x @ beta_global
    global_scale = max(float(np.sqrt(np.mean(global_resid ** 2))), 1e-6)

    beta = np.tile(beta_global, (n_classes, 1))
    scales = np.full(n_classes, global_scale)

    for c in range(n_classes):
        mask = pred == c
        if np.sum(mask) >= 6:
            xc = np.column_stack([np.ones(np.sum(mask)), x[mask]])
            yc = y[mask]
            bc = np.linalg.solve(
                xc.T @ xc + 1e-5 * np.eye(2),
                xc.T @ yc,
            )
            resid = yc - xc @ bc
            beta[c] = bc
            scales[c] = max(float(np.sqrt(np.mean(resid ** 2))), 1e-6)

    expected = beta[epred, 0] + beta[epred, 1] * ex
    scores = np.abs(ey - expected) / scales[epred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)

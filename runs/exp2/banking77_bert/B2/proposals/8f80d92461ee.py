import numpy as np

FAMILY = "logit"
ARM = "edl"
RATIONALE = (
    "OOS inputs should often produce a more uniform Dirichlet mean across competing intents than "
    "in-scope inputs, even at similar total concentration. Class-conditional train calibration of "
    "Dirichlet-mean entropy isolates this concentration-shape ambiguity from vacuity."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    def shape_features(logits):
        alpha = np.maximum(ctx.h.alpha(logits).astype(float), eps)
        prob = alpha / np.sum(alpha, axis=1, keepdims=True)

        entropy = -np.sum(prob * np.log(np.maximum(prob, eps)), axis=1)
        entropy /= max(np.log(prob.shape[1]), eps)

        top2 = np.partition(prob, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        concentration = np.log(np.sum(alpha, axis=1))

        return entropy, margin, concentration, np.argmax(prob, axis=1)

    tr_entropy, tr_margin, tr_conc, tr_pred = shape_features(tr.logits)
    ev_entropy, ev_margin, ev_conc, ev_pred = shape_features(ev.logits)

    k = tr.logits.shape[1]
    counts = np.bincount(tr_pred, minlength=k).astype(float)

    global_x = np.column_stack((tr_margin, tr_conc))
    global_y = tr_entropy
    global_design = np.column_stack((np.ones(len(tr)), global_x))
    ridge = np.diag([1e-6, 1e-4, 1e-4])
    global_beta = np.linalg.solve(
        global_design.T @ global_design + ridge,
        global_design.T @ global_y,
    )
    global_resid = global_y - global_design @ global_beta
    global_scale = max(float(np.std(global_resid)), 1e-4)

    betas = np.tile(global_beta, (k, 1))
    scales = np.full(k, global_scale)

    for c in range(k):
        mask = tr_pred == c
        if counts[c] >= 8:
            x = np.column_stack((np.ones(int(counts[c])), tr_margin[mask], tr_conc[mask]))
            y = tr_entropy[mask]
            beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
            resid = y - x @ beta
            betas[c] = beta
            scales[c] = max(float(np.std(resid)), 1e-4)

    ev_design = np.column_stack((np.ones(len(ev)), ev_margin, ev_conc))
    expected_entropy = np.sum(ev_design * betas[ev_pred], axis=1)
    score_values = (ev_entropy - expected_entropy) / scales[ev_pred]

    # Retain a tiny, bounded shape term so otherwise nearly tied calibrated residuals
    # remain ordered without using any evaluation-set aggregate.
    score_values += 1e-5 * (ev_entropy - ev_margin)
    return np.nan_to_num(score_values, nan=0.0, posinf=1e6, neginf=-1e6)

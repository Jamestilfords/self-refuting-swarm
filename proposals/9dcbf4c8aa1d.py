import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A classifier-selected intent should have both low shrunk diagonal Gaussian surprise and better heteroscedastic feature support than its logit runner-up. This flags OOS queries that are globally unusual for the predicted intent or whose representation contradicts the classifier's intent choice."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0094", "m_0254"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = np.asarray(tr.pooled, dtype=float)
    xev = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = xtr.shape[1]

    global_mean = np.mean(xtr, axis=0)
    global_var = np.mean((xtr - global_mean) ** 2, axis=0)
    floor = max(float(np.mean(global_var)) * 1e-6, 1e-10)
    global_var = np.maximum(global_var, floor)

    means = np.empty((k, d), dtype=float)
    variances = np.empty((k, d), dtype=float)
    for c in range(k):
        xc = xtr[y == c]
        n_c = len(xc)
        if n_c == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            mu = np.mean(xc, axis=0)
            raw_var = np.mean((xc - mu) ** 2, axis=0)
            shrink = n_c / (n_c + 32.0)
            means[c] = mu
            variances[c] = np.maximum(
                shrink * raw_var + (1.0 - shrink) * global_var,
                floor,
            )

    logdet = np.sum(np.log(variances), axis=1)

    def diagonal_nll(x, classes):
        delta = x - means[classes]
        return np.sum((delta * delta) / variances[classes], axis=1) + logdet[classes]

    def predicted_and_runner(logits):
        pred = np.argmax(logits, axis=1)
        if logits.shape[1] < 2:
            return pred, pred
        masked = np.array(logits, copy=True)
        masked[np.arange(len(pred)), pred] = -np.inf
        return pred, np.argmax(masked, axis=1)

    tr_pred, tr_runner = predicted_and_runner(np.asarray(tr.logits))
    tr_self = diagonal_nll(xtr, tr_pred)
    tr_runner_nll = diagonal_nll(xtr, tr_runner)
    tr_disagreement = tr_self - tr_runner_nll

    radial_center = np.zeros(k, dtype=float)
    radial_scale = np.ones(k, dtype=float)
    disagreement_center = np.zeros(k, dtype=float)
    disagreement_scale = np.ones(k, dtype=float)
    for c in range(k):
        mask = tr_pred == c
        if np.any(mask):
            radial_center[c] = np.mean(tr_self[mask])
            radial_scale[c] = max(float(np.std(tr_self[mask])), 1e-6)
            disagreement_center[c] = np.mean(tr_disagreement[mask])
            disagreement_scale[c] = max(float(np.std(tr_disagreement[mask])), 1e-6)

    ev_pred, ev_runner = predicted_and_runner(np.asarray(ev.logits))
    ev_self = diagonal_nll(xev, ev_pred)
    ev_runner_nll = diagonal_nll(xev, ev_runner)
    ev_disagreement = ev_self - ev_runner_nll

    radial_surprise = (ev_self - radial_center[ev_pred]) / radial_scale[ev_pred]
    rival_advantage = (
        (ev_disagreement - disagreement_center[ev_pred])
        / disagreement_scale[ev_pred]
    )
    scores = radial_surprise + 0.5 * np.maximum(rival_advantage, 0.0)
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

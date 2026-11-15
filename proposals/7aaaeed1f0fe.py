import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "For each classifier-selected intent, in-scope logits exhibit a learned relationship "
    "between overall centered logit magnitude and the decisive top-two margin. Queries "
    "whose margin is unexpectedly small for their logit scale are suspicious."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0008"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def logit_features(logits):
        z = np.asarray(logits, dtype=float)
        centered = z - np.mean(z, axis=1, keepdims=True)
        norm = np.sqrt(np.sum(centered * centered, axis=1))
        top2 = np.partition(z, -2, axis=1)[:, -2:]
        margin = np.max(top2, axis=1) - np.min(top2, axis=1)
        return norm, margin

    tr_norm, tr_margin = logit_features(tr.logits)
    ev_norm, ev_margin = logit_features(ev.logits)
    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)

    def fit_line(x, y):
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        xc = x - x_mean
        denom = float(np.dot(xc, xc)) + 1e-3 * max(len(x), 1)
        slope = float(np.dot(xc, y - y_mean) / denom)
        intercept = y_mean - slope * x_mean
        resid = y - (intercept + slope * x)
        scale = float(np.sqrt(np.mean(resid * resid) + 1e-6))
        return intercept, slope, scale

    global_fit = fit_line(tr_norm, tr_margin)
    scores = np.empty(len(ev_margin), dtype=float)

    for c in np.unique(ev_pred):
        mask_tr = tr_pred == c
        if np.sum(mask_tr) >= 8:
            intercept, slope, scale = fit_line(
                tr_norm[mask_tr], tr_margin[mask_tr]
            )
        else:
            intercept, slope, scale = global_fit

        mask_ev = ev_pred == c
        expected_margin = intercept + slope * ev_norm[mask_ev]
        scores[mask_ev] = (expected_margin - ev_margin[mask_ev]) / scale

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

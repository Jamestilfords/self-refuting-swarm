import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "OOS queries often have both a smaller top-two class margin and reduced centered-logit "
    "dispersion than training examples assigned to the same predicted intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape_features(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        dispersion = np.sqrt(np.mean(centered * centered, axis=1))
        return margin, dispersion, np.argmax(logits, axis=1)

    tr_margin, tr_dispersion, tr_pred = shape_features(tr.logits)
    ev_margin, ev_dispersion, ev_pred = shape_features(ev.logits)

    k = tr.logits.shape[1]
    counts = np.bincount(tr_pred, minlength=k).astype(float)

    global_margin_mean = np.mean(tr_margin)
    global_disp_mean = np.mean(tr_dispersion)
    global_margin_std = max(np.std(tr_margin), 1e-6)
    global_disp_std = max(np.std(tr_dispersion), 1e-6)

    margin_mean = np.full(k, global_margin_mean)
    disp_mean = np.full(k, global_disp_mean)
    margin_std = np.full(k, global_margin_std)
    disp_std = np.full(k, global_disp_std)

    for c in range(k):
        mask = tr_pred == c
        if counts[c] >= 5:
            margin_mean[c] = np.mean(tr_margin[mask])
            disp_mean[c] = np.mean(tr_dispersion[mask])
            margin_std[c] = max(np.std(tr_margin[mask]), 1e-6)
            disp_std[c] = max(np.std(tr_dispersion[mask]), 1e-6)

    margin_deficit = (margin_mean[ev_pred] - ev_margin) / margin_std[ev_pred]
    dispersion_deficit = (disp_mean[ev_pred] - ev_dispersion) / disp_std[ev_pred]
    scores = margin_deficit + 0.35 * dispersion_deficit
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

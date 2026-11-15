import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "OOS queries often produce logits with unusually weak overall magnitude and decision separation relative to in-scope examples assigned to the same predicted intent. Class-conditional calibration prevents naturally low-scale intent classes from being treated as OOS by default."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr_logits = np.asarray(ctx.train.logits, dtype=float)
    ev_logits = np.asarray(ctx.eval.logits, dtype=float)

    def shape_features(logits):
        norm = np.sqrt(np.sum(logits * logits, axis=1))
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        return norm, margin

    tr_norm, tr_margin = shape_features(tr_logits)
    ev_norm, ev_margin = shape_features(ev_logits)
    tr_pred = np.argmax(tr_logits, axis=1)
    ev_pred = np.argmax(ev_logits, axis=1)

    global_norm_mean = np.mean(tr_norm)
    global_margin_mean = np.mean(tr_margin)
    global_norm_std = max(np.std(tr_norm), 1e-6)
    global_margin_std = max(np.std(tr_margin), 1e-6)

    n_classes = tr_logits.shape[1]
    norm_mean = np.full(n_classes, global_norm_mean)
    margin_mean = np.full(n_classes, global_margin_mean)
    norm_std = np.full(n_classes, global_norm_std)
    margin_std = np.full(n_classes, global_margin_std)

    for c in range(n_classes):
        mask = tr_pred == c
        if np.sum(mask) >= 5:
            norm_mean[c] = np.mean(tr_norm[mask])
            margin_mean[c] = np.mean(tr_margin[mask])
            norm_std[c] = max(np.std(tr_norm[mask]), 1e-6)
            margin_std[c] = max(np.std(tr_margin[mask]), 1e-6)

    norm_deficit = (norm_mean[ev_pred] - ev_norm) / norm_std[ev_pred]
    margin_deficit = (margin_mean[ev_pred] - ev_margin) / margin_std[ev_pred]
    return norm_deficit + 0.15 * margin_deficit

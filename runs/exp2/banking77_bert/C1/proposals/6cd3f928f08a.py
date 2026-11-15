import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope examples tend to have class-conditional logit shapes with a typical top-two margin and centered-logit norm. OOS queries often have both reduced winner separation and atypical overall logit concentration."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape_features(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        centered = logits - logits.mean(axis=1, keepdims=True)
        norm = np.sqrt(np.mean(centered * centered, axis=1))
        return margin, norm

    tr_margin, tr_norm = shape_features(tr.logits)
    ev_margin, ev_norm = shape_features(ev.logits)

    k = tr.logits.shape[1]
    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)

    global_margin_mean = np.mean(tr_margin)
    global_norm_mean = np.mean(tr_norm)
    global_margin_std = np.std(tr_margin) + 1e-6
    global_norm_std = np.std(tr_norm) + 1e-6

    margin_mean = np.full(k, global_margin_mean)
    norm_mean = np.full(k, global_norm_mean)
    margin_std = np.full(k, global_margin_std)
    norm_std = np.full(k, global_norm_std)

    for c in range(k):
        mask = tr_pred == c
        if np.sum(mask) >= 5:
            cm = tr_margin[mask]
            cn = tr_norm[mask]
            margin_mean[c] = np.mean(cm)
            norm_mean[c] = np.mean(cn)
            margin_std[c] = np.std(cm) + 1e-6
            norm_std[c] = np.std(cn) + 1e-6

    margin_deficit = (margin_mean[ev_pred] - ev_margin) / margin_std[ev_pred]
    norm_deviation = np.abs(ev_norm - norm_mean[ev_pred]) / norm_std[ev_pred]

    return margin_deficit + 0.35 * norm_deviation

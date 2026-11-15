import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Low classifier confidence and large distance from the classifier-selected intent "
    "prototype are complementary OOS cues. Predicted-intent-conditional empirical tail "
    "maps fitted on training data put both signals on a common, class-aware scale."
)
SOURCE = ""
PARENTS = ["msp", "d_port"]
MEMORY_USED = ["m_0010", "m_0037", "m_0099"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = np.asarray(tr.feats, dtype=np.float64)
    xev = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    means = np.zeros((k, xtr.shape[1]), dtype=np.float64)
    counts = np.bincount(y, minlength=k)
    global_mean = np.mean(xtr, axis=0)
    for c in range(k):
        if counts[c] > 0:
            means[c] = np.mean(xtr[y == c], axis=0)
        else:
            means[c] = global_mean

    residuals = xtr - means[y]
    var = np.mean(residuals * residuals, axis=0)
    floor = max(float(np.median(var)) * 1e-3, 1e-8)
    scale = 1.0 / np.sqrt(var + floor)

    dtr = np.mean(((xtr - means[pred_tr]) * scale) ** 2, axis=1)
    dev = np.mean(((xev - means[pred_ev]) * scale) ** 2, axis=1)

    ltr = np.asarray(ctx.h.msp(tr.logits), dtype=np.float64)
    lev = np.asarray(ctx.h.msp(ev.logits), dtype=np.float64)

    def conditional_tail(train_values, train_classes, test_values, test_classes):
        out = np.empty(test_values.shape[0], dtype=np.float64)
        global_sorted = np.sort(train_values)
        for c in range(k):
            test_mask = test_classes == c
            if not np.any(test_mask):
                continue
            ref = train_values[train_classes == c]
            if ref.size < 8:
                ref = global_sorted
            else:
                ref = np.sort(ref)
            ranks = np.searchsorted(ref, test_values[test_mask], side="right")
            out[test_mask] = ranks / (ref.size + 1.0)
        return out

    msp_tail = conditional_tail(ltr, pred_tr, lev, pred_ev)
    dist_tail = conditional_tail(dtr, pred_tr, dev, pred_ev)

    return 0.5 * (msp_tail + dist_tail)

import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "OOS queries often have both a reduced top-two logit margin and unusually low "
    "centered-logit concentration; class-conditional train calibration accounts for intent-specific logit scales."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0011"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    def features(logits):
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        concentration = np.sqrt(np.mean(centered * centered, axis=1))
        return margin, concentration

    tr_margin, tr_conc = features(tr.logits)
    ev_margin, ev_conc = features(ev.logits)
    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    n_classes = tr.logits.shape[1]
    score_out = np.empty(ev.logits.shape[0], dtype=float)

    global_margin_med = np.median(tr_margin)
    global_margin_scale = np.median(np.abs(tr_margin - global_margin_med)) + eps
    global_conc_med = np.median(tr_conc)
    global_conc_scale = np.median(np.abs(tr_conc - global_conc_med)) + eps

    for c in range(n_classes):
        train_mask = pred_tr == c
        eval_mask = pred_ev == c
        if not np.any(eval_mask):
            continue

        if np.sum(train_mask) >= 5:
            m = tr_margin[train_mask]
            z = tr_conc[train_mask]
            m_med = np.median(m)
            z_med = np.median(z)
            m_scale = np.median(np.abs(m - m_med)) + eps
            z_scale = np.median(np.abs(z - z_med)) + eps
        else:
            m_med, m_scale = global_margin_med, global_margin_scale
            z_med, z_scale = global_conc_med, global_conc_scale

        margin_deficit = (m_med - ev_margin[eval_mask]) / m_scale
        concentration_deficit = (z_med - ev_conc[eval_mask]) / z_scale
        score_out[eval_mask] = margin_deficit + concentration_deficit

    return np.nan_to_num(score_out, nan=0.0, posinf=1e6, neginf=-1e6)

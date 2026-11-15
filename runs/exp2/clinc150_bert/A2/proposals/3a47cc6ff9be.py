import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope predictions typically have class-specific logit concentration and a stable top-two separation. Queries with unusually diffuse, low-margin logits relative to training examples assigned to the same predicted class are more likely to be out-of-scope."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape_values(logits):
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        concentration = np.sqrt(np.mean(centered * centered, axis=1))
        pred = np.argmax(logits, axis=1)
        return margin, concentration, pred

    tr_margin, tr_conc, tr_pred = shape_values(tr.logits)
    ev_margin, ev_conc, ev_pred = shape_values(ev.logits)
    k = tr.logits.shape[1]

    global_m_mean = np.mean(tr_margin)
    global_c_mean = np.mean(tr_conc)
    global_m_std = max(np.std(tr_margin), 1e-6)
    global_c_std = max(np.std(tr_conc), 1e-6)

    m_mean = np.full(k, global_m_mean)
    c_mean = np.full(k, global_c_mean)
    m_std = np.full(k, global_m_std)
    c_std = np.full(k, global_c_std)

    for cls in range(k):
        mask = tr_pred == cls
        if np.sum(mask) >= 2:
            m_mean[cls] = np.mean(tr_margin[mask])
            c_mean[cls] = np.mean(tr_conc[mask])
            m_std[cls] = max(np.std(tr_margin[mask]), 1e-6)
            c_std[cls] = max(np.std(tr_conc[mask]), 1e-6)

    margin_deficit = (m_mean[ev_pred] - ev_margin) / m_std[ev_pred]
    concentration_deficit = (c_mean[ev_pred] - ev_conc) / c_std[ev_pred]
    return np.clip(0.5 * (margin_deficit + concentration_deficit), -1e6, 1e6)

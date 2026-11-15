import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Class-conditional density percentiles measure whether a query is unusually far from the intent predicted by the classifier, rather than only from its nearest Gaussian class. Bottleneck and encoder-space percentiles are additively fused with train-fitted stability weights, retaining sensitivity to both near-intent and broad-domain OOS shifts."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = int(np.max(labels)) + 1

    g_port = ctx.h.gda(tr.feats, labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, labels, cov="tied")

    tr_port_all = g_port.class_mahalanobis_sq(tr.feats)
    tr_full_all = g_full.class_mahalanobis_sq(tr.pooled)
    ev_port_all = g_port.class_mahalanobis_sq(ev.feats)
    ev_full_all = g_full.class_mahalanobis_sq(ev.pooled)

    tr_idx = np.arange(labels.shape[0])
    tr_port = tr_port_all[tr_idx, labels]
    tr_full = tr_full_all[tr_idx, labels]

    pred = np.argmax(ev.logits, axis=1).astype(int)
    pred = np.clip(pred, 0, n_classes - 1)
    ev_idx = np.arange(pred.shape[0])
    ev_port = ev_port_all[ev_idx, pred]
    ev_full = ev_full_all[ev_idx, pred]

    port_score = np.empty(ev_port.shape[0], dtype=float)
    full_score = np.empty(ev_full.shape[0], dtype=float)

    port_stability = []
    full_stability = []
    for c in range(n_classes):
        mask = labels == c
        a = np.sort(tr_port[mask])
        b = np.sort(tr_full[mask])

        if a.size < 2:
            a = np.sort(tr_port)
            b = np.sort(tr_full)

        take = pred == c
        if np.any(take):
            port_score[take] = np.searchsorted(a, ev_port[take], side="right") / float(a.size)
            full_score[take] = np.searchsorted(b, ev_full[take], side="right") / float(b.size)

        if a.size >= 6:
            odd_a = a[::2]
            even_a = a[1::2]
            odd_b = b[::2]
            even_b = b[1::2]
            q = np.linspace(0.1, 0.9, 9)
            port_stability.append(
                np.mean(np.abs(np.quantile(odd_a, q) - np.quantile(even_a, q)))
                / max(np.std(a), 1e-6)
            )
            full_stability.append(
                np.mean(np.abs(np.quantile(odd_b, q) - np.quantile(even_b, q)))
                / max(np.std(b), 1e-6)
            )

    s_port = max(float(np.median(port_stability)) if port_stability else 1.0, 1e-4)
    s_full = max(float(np.median(full_stability)) if full_stability else 1.0, 1e-4)
    w_port = 1.0 / s_port
    w_full = 1.0 / s_full
    denom = w_port + w_full
    w_port /= denom
    w_full /= denom

    scores = w_port * port_score + w_full * full_score
    return np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)

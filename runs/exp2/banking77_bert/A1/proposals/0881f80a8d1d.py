import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "An in-scope query should have a favorable cosine margin between the prototype of its classifier-predicted intent and competing intent prototypes. Calibrating this margin within predicted intents makes unusually weak directional commitment more suspicious without reducing to maximum prototype similarity."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0121"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_tr = np.asarray(tr.feats, dtype=float)
    x_ev = np.asarray(ev.feats, dtype=float)
    y_tr = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]

    tr_unit = x_tr / np.maximum(np.linalg.norm(x_tr, axis=1, keepdims=True), 1e-12)
    ev_unit = x_ev / np.maximum(np.linalg.norm(x_ev, axis=1, keepdims=True), 1e-12)

    prototypes = np.zeros((k, tr_unit.shape[1]), dtype=float)
    counts = np.bincount(y_tr, minlength=k).astype(float)
    np.add.at(prototypes, y_tr, tr_unit)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)

    tr_sim = tr_unit @ prototypes.T
    ev_sim = ev_unit @ prototypes.T
    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)

    if k > 1:
        tr_own = tr_sim[np.arange(tr_sim.shape[0]), tr_pred]
        ev_own = ev_sim[np.arange(ev_sim.shape[0]), ev_pred]

        tr_other = tr_sim.copy()
        ev_other = ev_sim.copy()
        tr_other[np.arange(tr_other.shape[0]), tr_pred] = -np.inf
        ev_other[np.arange(ev_other.shape[0]), ev_pred] = -np.inf

        tr_margin = tr_own - np.max(tr_other, axis=1)
        ev_margin = ev_own - np.max(ev_other, axis=1)
    else:
        tr_margin = tr_sim[:, 0]
        ev_margin = ev_sim[:, 0]

    score_out = np.empty(ev_margin.shape[0], dtype=float)
    fallback = np.sort(tr_margin)

    for c in range(k):
        mask_ev = ev_pred == c
        if not np.any(mask_ev):
            continue
        reference = tr_margin[tr_pred == c]
        if reference.size < 8:
            reference = fallback
        reference = np.sort(reference)
        rank = np.searchsorted(reference, ev_margin[mask_ev], side="right")
        tail = 1.0 - rank / (reference.size + 1.0)
        score_out[mask_ev] = tail + 1e-3 * (1.0 - np.clip(ev_margin[mask_ev], -2.0, 2.0) / 2.0)

    return np.nan_to_num(score_out, nan=0.0, posinf=1.0, neginf=0.0)

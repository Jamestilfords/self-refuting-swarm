import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "An in-scope item assigned to an intent should retain a geometric advantage for that intent "
    "over nearby alternative prototypes. Class- and rival-conditioned calibration of selected-class "
    "disadvantage and nearest-versus-second-nearest ambiguity flags classifier-geometry disagreement."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0032", "m_0211", "m_0112"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    tr_logits = np.asarray(tr.logits, dtype=float)
    ev_logits = np.asarray(ev.logits, dtype=float)
    tr_pred = np.argmax(tr_logits, axis=1)
    ev_pred = np.argmax(ev_logits, axis=1)
    n_classes = tr_logits.shape[1]

    tr_d = np.asarray(g.class_mahalanobis_sq(tr.pooled), dtype=float)
    ev_d = np.asarray(g.class_mahalanobis_sq(ev.pooled), dtype=float)

    def geometry(dist, pred):
        n, k = dist.shape
        chosen = dist[np.arange(n), pred]
        masked = dist.copy()
        masked[np.arange(n), pred] = np.inf
        rival = np.argmin(masked, axis=1)
        nearest_other = masked[np.arange(n), rival]

        if k > 2:
            two = np.partition(masked, 1, axis=1)[:, :2]
            second_other = two[:, 1]
            ambiguity = nearest_other - second_other
        else:
            ambiguity = np.zeros(n, dtype=float)

        disadvantage = chosen - nearest_other
        return disadvantage, ambiguity, rival

    tr_disadv, tr_ambig, tr_rival = geometry(tr_d, tr_pred)
    ev_disadv, ev_ambig, ev_rival = geometry(ev_d, ev_pred)

    # Higher disadvantage is worse; ambiguity is negative when the two nearest
    # competing intents are close, so its negative is an OOS-oriented feature.
    tr_features = np.column_stack([tr_disadv, -tr_ambig])
    ev_features = np.column_stack([ev_disadv, -ev_ambig])

    def empirical_tail(reference, values):
        ref = np.sort(np.asarray(reference, dtype=float))
        if ref.size == 0:
            return np.full(values.shape[0], 0.5, dtype=float)
        ranks = np.searchsorted(ref, values, side="right")
        return ranks.astype(float) / (ref.size + 1.0)

    global_scores = np.empty((len(ev_pred), 2), dtype=float)

    for j in range(2):
        global_scores[:, j] = empirical_tail(tr_features[:, j], ev_features[:, j])

    out = np.empty((len(ev_pred), 2), dtype=float)
    for i in range(len(ev_pred)):
        pair_mask = (tr_pred == ev_pred[i]) & (tr_rival == ev_rival[i])
        class_mask = tr_pred == ev_pred[i]

        # Ordered predicted/nearest-rival geometry is preferred when sufficiently
        # represented; otherwise retain robust predicted-intent calibration.
        if np.sum(pair_mask) >= 12:
            ref_mask = pair_mask
        elif np.sum(class_mask) >= 12:
            ref_mask = class_mask
        else:
            ref_mask = np.ones(len(tr_pred), dtype=bool)

        out[i, 0] = empirical_tail(tr_features[ref_mask, 0], ev_features[i:i + 1, 0])[0]
        out[i, 1] = empirical_tail(tr_features[ref_mask, 1], ev_features[i:i + 1, 1])[0]

    scores = 0.7 * out[:, 0] + 0.3 * out[:, 1]
    return np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)

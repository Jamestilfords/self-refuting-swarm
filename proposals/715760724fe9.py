import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The Lorenz curve of rival Dirichlet evidence captures how evidence is concentrated across competing intents, independently of total evidence. "
    "Class-conditioned deviations from the usual rival-evidence inequality profile can expose atypical evidential competition for OOS inputs."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0214"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def lorenz_features(logits, alpha):
        alpha = np.nan_to_num(alpha, nan=1.0, posinf=1e12, neginf=1e-12)
        alpha = np.maximum(alpha, 1e-12)
        n, k = alpha.shape
        pred = np.argmax(logits, axis=1)

        total = np.sum(alpha, axis=1, keepdims=True)
        share = alpha / np.maximum(total, 1e-12)

        rival = share.copy()
        rival[np.arange(n), pred] = 0.0
        rival_total = np.sum(rival, axis=1, keepdims=True)
        rival = rival / np.maximum(rival_total, 1e-12)

        ordered = np.sort(rival, axis=1)[:, ::-1]
        ranks = np.unique(np.minimum(np.array([1, 2, 4, 8, 16, 32]), k - 1))
        cumulative = np.cumsum(ordered, axis=1)
        curve = cumulative[:, ranks - 1]

        winner_share = share[np.arange(n), pred][:, None]
        return pred, np.concatenate([winner_share, curve], axis=1)

    tr_pred, tr_feat = lorenz_features(tr.logits, ctx.h.alpha(tr.logits))
    ev_pred, ev_feat = lorenz_features(ev.logits, ctx.h.alpha(ev.logits))

    global_med = np.median(tr_feat, axis=0)
    global_mad = np.median(np.abs(tr_feat - global_med), axis=0)
    global_scale = np.maximum(1.4826 * global_mad, 1e-4)

    k = tr.logits.shape[1]
    medians = np.tile(global_med, (k, 1))
    scales = np.tile(global_scale, (k, 1))

    for c in range(k):
        members = tr_feat[tr_pred == c]
        if members.shape[0] >= 8:
            med = np.median(members, axis=0)
            mad = np.median(np.abs(members - med), axis=0)
            medians[c] = med
            scales[c] = np.maximum(1.4826 * mad, 1e-4)

    z = (ev_feat - medians[ev_pred]) / scales[ev_pred]
    return np.sum(np.minimum(z * z, 100.0), axis=1)

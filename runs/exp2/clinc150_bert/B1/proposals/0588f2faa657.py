import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Class-conditional empirical density tails make bottleneck and encoder distances comparable without relying on "
    "evaluation statistics. Additively combining their train-fitted reliability-weighted surprise scores captures "
    "near-OOS decision-feature deviations and broader semantic-domain deviations."
)
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = int(np.max(labels)) + 1

    g_port = ctx.h.gda(tr.feats, labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, labels, cov="tied")

    port_train_all = np.asarray(g_port.class_mahalanobis_sq(tr.feats), dtype=float)
    full_train_all = np.asarray(g_full.class_mahalanobis_sq(tr.pooled), dtype=float)
    port_eval_all = np.asarray(g_port.class_mahalanobis_sq(ev.feats), dtype=float)
    full_eval_all = np.asarray(g_full.class_mahalanobis_sq(ev.pooled), dtype=float)

    train_rows = np.arange(labels.shape[0])
    port_train_own = port_train_all[train_rows, labels]
    full_train_own = full_train_all[train_rows, labels]

    port_eval_class = np.argmin(port_eval_all, axis=1)
    full_eval_class = np.argmin(full_eval_all, axis=1)
    eval_rows = np.arange(port_eval_all.shape[0])
    port_eval_min = port_eval_all[eval_rows, port_eval_class]
    full_eval_min = full_eval_all[eval_rows, full_eval_class]

    port_scale = np.std(port_train_own)
    full_scale = np.std(full_train_own)
    port_weight = 1.0 / max(float(port_scale), 1e-8)
    full_weight = 1.0 / max(float(full_scale), 1e-8)
    total_weight = port_weight + full_weight
    port_weight /= total_weight
    full_weight /= total_weight

    port_score = np.empty_like(port_eval_min)
    full_score = np.empty_like(full_eval_min)

    for c in range(n_classes):
        port_ref = np.sort(port_train_own[labels == c])
        full_ref = np.sort(full_train_own[labels == c])

        port_mask = port_eval_class == c
        if np.any(port_mask):
            values = port_eval_min[port_mask]
            exceed = port_ref.size - np.searchsorted(port_ref, values, side="right")
            tail_prob = (exceed + 1.0) / (port_ref.size + 1.0)
            port_score[port_mask] = -np.log(tail_prob)

        full_mask = full_eval_class == c
        if np.any(full_mask):
            values = full_eval_min[full_mask]
            exceed = full_ref.size - np.searchsorted(full_ref, values, side="right")
            tail_prob = (exceed + 1.0) / (full_ref.size + 1.0)
            full_score[full_mask] = -np.log(tail_prob)

    scores = port_weight * port_score + full_weight * full_score
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)

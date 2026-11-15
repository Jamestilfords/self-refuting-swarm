import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Bottleneck and encoder densities are converted to predicted-intent-conditional empirical "
    "tail probabilities, then additively fused with train-fitted view reliabilities. This preserves "
    "near-OOS sensitivity from the bottleneck while allowing encoder-space density to dominate where it better supports classifier assignments."
)
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_port = np.asarray(g_port.neg_log_density(tr.feats), dtype=float)
    train_full = np.asarray(g_full.neg_log_density(tr.pooled), dtype=float)
    eval_port = np.asarray(g_port.neg_log_density(ev.feats), dtype=float)
    eval_full = np.asarray(g_full.neg_log_density(ev.pooled), dtype=float)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    port_nearest = np.argmin(g_port.class_mahalanobis_sq(tr.feats), axis=1)
    full_nearest = np.argmin(g_full.class_mahalanobis_sq(tr.pooled), axis=1)

    global_port = np.sort(train_port[np.isfinite(train_port)])
    global_full = np.sort(train_full[np.isfinite(train_full)])
    if global_port.size == 0:
        global_port = np.array([0.0])
    if global_full.size == 0:
        global_full = np.array([0.0])

    port_tables = []
    full_tables = []
    weights = np.empty(n_classes, dtype=float)

    for cls in range(n_classes):
        mask = train_pred == cls

        port_values = train_port[mask]
        port_values = port_values[np.isfinite(port_values)]
        if port_values.size < 8:
            port_values = global_port
        else:
            port_values = np.sort(port_values)

        full_values = train_full[mask]
        full_values = full_values[np.isfinite(full_values)]
        if full_values.size < 8:
            full_values = global_full
        else:
            full_values = np.sort(full_values)

        port_tables.append(port_values)
        full_tables.append(full_values)

        if np.any(mask):
            port_agree = np.mean(port_nearest[mask] == cls)
            full_agree = np.mean(full_nearest[mask] == cls)
            port_rel = (port_agree * mask.sum() + 1.0) / (mask.sum() + 2.0)
            full_rel = (full_agree * mask.sum() + 1.0) / (mask.sum() + 2.0)
        else:
            port_rel = 0.5
            full_rel = 0.5

        weights[cls] = port_rel / (port_rel + full_rel)

    def upper_tail(values, predictions, tables):
        out = np.empty(values.shape[0], dtype=float)
        for cls in range(n_classes):
            mask = predictions == cls
            if not np.any(mask):
                continue
            reference = tables[cls]
            ranks = np.searchsorted(reference, values[mask], side="right")
            out[mask] = ranks / float(reference.size + 1)
        return np.clip(out, 0.0, 1.0)

    port_tail = upper_tail(eval_port, eval_pred, port_tables)
    full_tail = upper_tail(eval_full, eval_pred, full_tables)

    port_weight = weights[eval_pred]
    scores = port_weight * port_tail + (1.0 - port_weight) * full_tail
    return np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)

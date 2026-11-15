import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck and pooled-space density scores capture complementary near- and far-OOS deviations. Each density is calibrated by its train-set empirical tail probability, then weighted inversely by its within-intent variability on training data."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    port_train = np.asarray(g_port.neg_log_density(tr.feats), dtype=float)
    full_train = np.asarray(g_full.neg_log_density(tr.pooled), dtype=float)
    port_eval = np.asarray(g_port.neg_log_density(ev.feats), dtype=float)
    full_eval = np.asarray(g_full.neg_log_density(ev.pooled), dtype=float)

    def tail_logit(train_scores, test_scores):
        finite_train = train_scores[np.isfinite(train_scores)]
        if finite_train.size == 0:
            finite_train = np.array([0.0])
        ordered = np.sort(finite_train)
        n = float(ordered.size)

        train_clean = np.nan_to_num(
            train_scores,
            nan=float(np.median(ordered)),
            posinf=float(ordered[-1]),
            neginf=float(ordered[0]),
        )
        test_clean = np.nan_to_num(
            test_scores,
            nan=float(np.median(ordered)),
            posinf=float(ordered[-1]),
            neginf=float(ordered[0]),
        )

        train_q = (np.searchsorted(ordered, train_clean, side="right") + 0.5) / (n + 1.0)
        test_q = (np.searchsorted(ordered, test_clean, side="right") + 0.5) / (n + 1.0)
        eps = 1.0 / (n + 2.0)

        train_q = np.clip(train_q, eps, 1.0 - eps)
        test_q = np.clip(test_q, eps, 1.0 - eps)
        return np.log(train_q / (1.0 - train_q)), np.log(test_q / (1.0 - test_q))

    port_train_t, port_eval_t = tail_logit(port_train, port_eval)
    full_train_t, full_eval_t = tail_logit(full_train, full_eval)

    labels = np.asarray(tr.labels)
    port_resid = np.empty_like(port_train_t)
    full_resid = np.empty_like(full_train_t)
    for label in np.unique(labels):
        mask = labels == label
        port_resid[mask] = port_train_t[mask] - np.mean(port_train_t[mask])
        full_resid[mask] = full_train_t[mask] - np.mean(full_train_t[mask])

    port_var = float(np.mean(port_resid * port_resid))
    full_var = float(np.mean(full_resid * full_resid))
    w_port = 1.0 / max(port_var, 1e-8)
    w_full = 1.0 / max(full_var, 1e-8)
    total = w_port + w_full

    return (w_port * port_eval_t + w_full * full_eval_t) / total

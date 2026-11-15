import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Class-conditional tail calibration makes bottleneck and encoder Mahalanobis densities comparable while preserving their different nearest-class assignments. Inverse-variance weights are fitted from training tail scores only, combining bottleneck sensitivity to near-OOS examples with encoder sensitivity to broader shifts."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def calibrated_tail(train_x, eval_x):
        g = ctx.h.gda(train_x, tr.labels, cov="tied", whiten=True)

        train_dmat = g.class_mahalanobis_sq(train_x)
        eval_dmat = g.class_mahalanobis_sq(eval_x)

        train_cls = np.argmin(train_dmat, axis=1)
        eval_cls = np.argmin(eval_dmat, axis=1)
        train_d = train_dmat[np.arange(train_dmat.shape[0]), train_cls]
        eval_d = eval_dmat[np.arange(eval_dmat.shape[0]), eval_cls]

        global_sorted = np.sort(train_d)
        train_tail = np.empty_like(train_d, dtype=float)
        eval_tail = np.empty_like(eval_d, dtype=float)

        n_classes = train_dmat.shape[1]
        for c in range(n_classes):
            mask_tr = train_cls == c
            mask_ev = eval_cls == c
            ref = train_d[mask_tr]
            if ref.size < 8:
                ref = global_sorted
            else:
                ref = np.sort(ref)

            denom = float(ref.size + 1)
            if np.any(mask_tr):
                ranks = np.searchsorted(ref, train_d[mask_tr], side="right")
                q = ranks / denom
                train_tail[mask_tr] = -np.log1p(-np.minimum(q, 1.0 - 1.0 / denom))
            if np.any(mask_ev):
                ranks = np.searchsorted(ref, eval_d[mask_ev], side="right")
                q = ranks / denom
                eval_tail[mask_ev] = -np.log1p(-np.minimum(q, 1.0 - 1.0 / denom))

        return train_tail, eval_tail

    port_train, port_eval = calibrated_tail(tr.feats, ev.feats)
    full_train, full_eval = calibrated_tail(tr.pooled, ev.pooled)

    port_var = max(float(np.var(port_train)), 1e-8)
    full_var = max(float(np.var(full_train)), 1e-8)
    w_port = 1.0 / port_var
    w_full = 1.0 / full_var
    weight_sum = w_port + w_full

    scores = (w_port * port_eval + w_full * full_eval) / weight_sum
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)

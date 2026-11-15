import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Class-mean directions capture the feature variation most useful for separating known intents. "
    "After removing that discriminative component, unusually large class-calibrated residual variation indicates unsupported semantic content."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0107", "m_0113"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_tr = np.asarray(tr.feats, dtype=np.float64)
    x_ev = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    n, d = x_tr.shape
    k = tr.logits.shape[1]
    mu = x_tr.mean(axis=0)

    means = np.empty((k, d), dtype=np.float64)
    for c in range(k):
        members = x_tr[y == c]
        means[c] = members.mean(axis=0) if len(members) else mu

    centered_means = means - mu
    _, singular, vt = np.linalg.svd(centered_means, full_matrices=False)
    if singular.size and singular[0] > 0:
        rank = int(np.sum(singular > singular[0] * 1e-8))
    else:
        rank = 0
    rank = min(rank, d - 1)

    if rank > 0:
        basis = vt[:rank].T
        tr_res = (x_tr - mu) - ((x_tr - mu) @ basis) @ basis.T
        ev_res = (x_ev - mu) - ((x_ev - mu) @ basis) @ basis.T
    else:
        tr_res = x_tr - mu
        ev_res = x_ev - mu

    global_var = np.var(tr_res, axis=0) + 1e-8
    class_var = np.empty((k, d), dtype=np.float64)
    for c in range(k):
        members = tr_res[y == c]
        if len(members) >= 2:
            class_var[c] = np.var(members, axis=0) + 0.05 * global_var + 1e-8
        else:
            class_var[c] = global_var

    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)

    train_raw = np.empty(n, dtype=np.float64)
    for c in range(k):
        mask = tr_pred == c
        if np.any(mask):
            train_raw[mask] = np.mean(
                (tr_res[mask] * tr_res[mask]) / class_var[c], axis=1
            )

    eval_raw = np.empty(len(x_ev), dtype=np.float64)
    for c in range(k):
        mask = ev_pred == c
        if np.any(mask):
            eval_raw[mask] = np.mean(
                (ev_res[mask] * ev_res[mask]) / class_var[c], axis=1
            )

    out = np.empty(len(x_ev), dtype=np.float64)
    for c in range(k):
        reference = np.sort(train_raw[tr_pred == c])
        mask = ev_pred == c
        if not np.any(mask):
            continue
        if len(reference) < 4:
            reference = np.sort(train_raw)
        out[mask] = np.searchsorted(reference, eval_raw[mask], side="right") / (
            len(reference) + 1.0
        )

    return np.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)

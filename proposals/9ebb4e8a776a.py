import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "Known intents tend to produce characteristic runner-up evidence competitors and "
    "winner-to-runner-up evidence gaps. Unusual rival identities or atypical gap sizes "
    "flag unfamiliar Dirichlet evidence-allocation patterns beyond total vacuity."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0180"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12
    k = tr.logits.shape[1]

    a_tr = np.maximum(np.asarray(ctx.h.alpha(tr.logits), dtype=float), eps)
    a_ev = np.maximum(np.asarray(ctx.h.alpha(ev.logits), dtype=float), eps)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    masked_tr = a_tr.copy()
    masked_tr[np.arange(masked_tr.shape[0]), pred_tr] = -np.inf
    rival_tr = np.argmax(masked_tr, axis=1)

    masked_ev = a_ev.copy()
    masked_ev[np.arange(masked_ev.shape[0]), pred_ev] = -np.inf
    rival_ev = np.argmax(masked_ev, axis=1)

    gap_tr = np.log(a_tr[np.arange(a_tr.shape[0]), pred_tr]) - np.log(
        a_tr[np.arange(a_tr.shape[0]), rival_tr]
    )
    gap_ev = np.log(a_ev[np.arange(a_ev.shape[0]), pred_ev]) - np.log(
        a_ev[np.arange(a_ev.shape[0]), rival_ev]
    )

    counts = np.ones((k, k), dtype=float)
    np.add.at(counts, (pred_tr, rival_tr), 1.0)
    counts[np.arange(k), np.arange(k)] = 0.0
    row_sum = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    rival_prob = counts / row_sum

    global_mean = float(np.mean(gap_tr))
    global_std = float(np.std(gap_tr))
    global_std = max(global_std, 1e-3)

    means = np.full(k, global_mean, dtype=float)
    scales = np.full(k, global_std, dtype=float)
    for c in range(k):
        vals = gap_tr[pred_tr == c]
        if vals.size:
            means[c] = (vals.sum() + 3.0 * global_mean) / (vals.size + 3.0)
            centered = vals - means[c]
            local_var = np.mean(centered * centered) if vals.size else global_std ** 2
            scales[c] = np.sqrt(
                (vals.size * local_var + 3.0 * global_std ** 2) / (vals.size + 3.0)
            )
    scales = np.maximum(scales, 1e-3)

    rival_surprise = -np.log(
        np.maximum(rival_prob[pred_ev, rival_ev], eps)
    )
    gap_surprise = np.abs((gap_ev - means[pred_ev]) / scales[pred_ev])

    return np.nan_to_num(
        rival_surprise + 0.5 * gap_surprise,
        nan=0.0,
        posinf=1e6,
        neginf=0.0,
    )

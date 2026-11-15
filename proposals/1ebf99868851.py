import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The rank ordering of Dirichlet concentrations captures which rival intents receive evidence, "
    "independently of total evidence magnitude. OOS queries can induce class-atypical competitor orderings "
    "even when vacuity or energy is in-distribution."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12
    k = tr.logits.shape[1]

    train_alpha = np.maximum(np.asarray(ctx.h.alpha(tr.logits), dtype=float), eps)
    eval_alpha = np.maximum(np.asarray(ctx.h.alpha(ev.logits), dtype=float), eps)

    train_share = train_alpha / np.maximum(train_alpha.sum(axis=1, keepdims=True), eps)
    eval_share = eval_alpha / np.maximum(eval_alpha.sum(axis=1, keepdims=True), eps)

    labels = np.asarray(tr.labels, dtype=int)
    sums = np.zeros((k, k), dtype=float)
    counts = np.bincount(labels, minlength=k).astype(float)
    np.add.at(sums, labels, train_share)

    global_proto = train_share.mean(axis=0)
    prototypes = sums / np.maximum(counts[:, None], 1.0)
    prototypes = np.where(counts[:, None] > 0, prototypes, global_proto[None, :])
    prototypes /= np.maximum(prototypes.sum(axis=1, keepdims=True), eps)

    prototype_order = np.argsort(-prototypes, axis=1)
    prototype_rank = np.empty_like(prototype_order)
    rank_values = np.broadcast_to(np.arange(k), (k, k))
    np.put_along_axis(prototype_rank, prototype_order, rank_values, axis=1)

    train_order = np.argsort(-train_share, axis=1)
    train_rank = np.empty_like(train_order)
    np.put_along_axis(
        train_rank,
        train_order,
        np.broadcast_to(np.arange(k), train_order.shape),
        axis=1,
    )

    train_proto_rank = prototype_rank[labels]
    train_weights = prototypes[labels]
    train_raw = np.sum(
        train_weights * np.abs(train_rank - train_proto_rank),
        axis=1,
    ) / max(k - 1, 1)

    global_med = np.median(train_raw)
    global_scale = np.median(np.abs(train_raw - global_med)) * 1.4826
    global_scale = max(float(global_scale), 1e-6)

    medians = np.full(k, global_med, dtype=float)
    scales = np.full(k, global_scale, dtype=float)
    for c in range(k):
        values = train_raw[labels == c]
        if values.size >= 3:
            center = np.median(values)
            spread = np.median(np.abs(values - center)) * 1.4826
            medians[c] = center
            scales[c] = max(float(spread), global_scale * 0.1, 1e-6)

    pred = np.argmax(ev.logits, axis=1)
    eval_order = np.argsort(-eval_share, axis=1)
    eval_rank = np.empty_like(eval_order)
    np.put_along_axis(
        eval_rank,
        eval_order,
        np.broadcast_to(np.arange(k), eval_order.shape),
        axis=1,
    )

    raw = np.sum(
        prototypes[pred] * np.abs(eval_rank - prototype_rank[pred]),
        axis=1,
    ) / max(k - 1, 1)

    scores = (raw - medians[pred]) / scales[pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

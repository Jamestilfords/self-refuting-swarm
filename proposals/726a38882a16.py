import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class residual directions are tightly constrained for supported intents. "
    "A class-conditioned random-hyperplane code measures whether an item's whitened trailing residual has a previously unsupported directional signature."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0188", "m_0287"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(np.float64)
    sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sums, y, x)
    means = sums / np.maximum(counts[:, None], 1.0)

    residuals = x - means[y]
    cov = (residuals.T @ residuals) / max(len(x) - k, 1)
    vals, vecs = np.linalg.eigh(cov)

    tail_dim = min(48, max(16, d // 12))
    tail_vals = np.maximum(vals[:tail_dim], 1e-8)
    tail_vecs = vecs[:, :tail_dim]
    scale = 1.0 / np.sqrt(tail_vals)

    train_tail = (residuals @ tail_vecs) * scale
    pred_eval = np.argmax(ev.logits, axis=1).astype(np.int64)
    eval_tail = ((z - means[pred_eval]) @ tail_vecs) * scale

    rng = np.random.default_rng(314159)
    planes = rng.standard_normal((tail_dim, 28))
    planes /= np.maximum(np.linalg.norm(planes, axis=0, keepdims=True), 1e-12)
    bit_weights = (1 << np.arange(28, dtype=np.uint32))

    train_codes = ((train_tail @ planes) >= 0).astype(np.uint32) @ bit_weights
    eval_codes = ((eval_tail @ planes) >= 0).astype(np.uint32) @ bit_weights

    byte_popcount = np.unpackbits(
        np.arange(256, dtype=np.uint8)[:, None], axis=1
    ).sum(axis=1).astype(np.uint8)

    def nearest_hamming(query_codes, ref_codes):
        out = np.empty(len(query_codes), dtype=np.float64)
        if len(ref_codes) == 0:
            out.fill(28.0)
            return out
        for start in range(0, len(query_codes), 256):
            q = query_codes[start:start + 256]
            xor = q[:, None] ^ ref_codes[None, :]
            dist = (
                byte_popcount[(xor & 255).astype(np.uint8)]
                + byte_popcount[((xor >> 8) & 255).astype(np.uint8)]
                + byte_popcount[((xor >> 16) & 255).astype(np.uint8)]
                + byte_popcount[((xor >> 24) & 255).astype(np.uint8)]
            )
            out[start:start + len(q)] = dist.min(axis=1)
        return out

    is_proto = (np.arange(len(x)) % 4) == 0
    global_refs = train_codes[is_proto]
    train_dist = np.empty(len(x), dtype=np.float64)
    eval_dist = np.empty(len(z), dtype=np.float64)

    for c in range(k):
        train_idx = np.flatnonzero(y == c)
        eval_idx = np.flatnonzero(pred_eval == c)
        if len(train_idx) == 0 and len(eval_idx) == 0:
            continue

        refs = train_idx[is_proto[train_idx]]
        if len(refs) < 2:
            ref_codes = global_refs
        else:
            ref_codes = train_codes[refs]

        if len(train_idx):
            train_dist[train_idx] = nearest_hamming(train_codes[train_idx], ref_codes)
        if len(eval_idx):
            eval_dist[eval_idx] = nearest_hamming(eval_codes[eval_idx], ref_codes)

    scores = np.empty(len(z), dtype=np.float64)
    for c in range(k):
        eval_idx = np.flatnonzero(pred_eval == c)
        if len(eval_idx) == 0:
            continue
        reference = np.sort(train_dist[y == c])
        if len(reference) < 4:
            reference = np.sort(train_dist)
        ranks = np.searchsorted(reference, eval_dist[eval_idx], side="right")
        scores[eval_idx] = ranks / max(len(reference), 1)

    radial = np.mean(eval_tail * eval_tail, axis=1)
    return scores + 1e-4 * np.log1p(radial)

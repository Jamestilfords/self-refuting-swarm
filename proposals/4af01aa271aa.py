import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "In-scope examples of each intent tend to have a characteristic centered logit direction, "
    "whereas OOS queries often produce a mismatched relative competition pattern even when confidence is high."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12
    k = tr.logits.shape[1]

    tr_centered = tr.logits - np.mean(tr.logits, axis=1, keepdims=True)
    tr_unit = tr_centered / np.maximum(
        np.linalg.norm(tr_centered, axis=1, keepdims=True), eps
    )

    sums = np.zeros((k, k), dtype=float)
    counts = np.zeros(k, dtype=float)
    np.add.at(sums, tr.labels, tr_unit)
    np.add.at(counts, tr.labels, 1.0)

    global_proto = np.mean(tr_unit, axis=0)
    global_proto /= max(np.linalg.norm(global_proto), eps)

    prototypes = np.tile(global_proto, (k, 1))
    present = counts > 0
    prototypes[present] = sums[present] / counts[present, None]
    prototypes /= np.maximum(
        np.linalg.norm(prototypes, axis=1, keepdims=True), eps
    )

    ev_centered = ev.logits - np.mean(ev.logits, axis=1, keepdims=True)
    ev_unit = ev_centered / np.maximum(
        np.linalg.norm(ev_centered, axis=1, keepdims=True), eps
    )
    predicted = np.argmax(ev.logits, axis=1)
    similarity = np.sum(ev_unit * prototypes[predicted], axis=1)

    return np.clip(1.0 - similarity, 0.0, 2.0)

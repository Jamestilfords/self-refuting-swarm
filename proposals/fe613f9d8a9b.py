import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Atypical centered-logit directions reveal unfamiliar winner-versus-rival competition patterns, "
    "while predicted-class Mahalanobis distance measures whether the representation is supported by that intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0200", "m_0201"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    def logit_direction_distance(train_logits, query_logits, train_pred, query_pred):
        train_centered = train_logits - train_logits.mean(axis=1, keepdims=True)
        query_centered = query_logits - query_logits.mean(axis=1, keepdims=True)
        train_unit = train_centered / np.maximum(
            np.linalg.norm(train_centered, axis=1, keepdims=True), eps
        )
        query_unit = query_centered / np.maximum(
            np.linalg.norm(query_centered, axis=1, keepdims=True), eps
        )

        n_classes = train_logits.shape[1]
        prototypes = np.zeros((n_classes, train_logits.shape[1]), dtype=float)
        global_proto = train_unit.mean(axis=0)
        global_proto /= max(np.linalg.norm(global_proto), eps)

        for c in range(n_classes):
            mask = train_pred == c
            if np.any(mask):
                proto = train_unit[mask].mean(axis=0)
                prototypes[c] = proto / max(np.linalg.norm(proto), eps)
            else:
                prototypes[c] = global_proto

        return 1.0 - np.sum(query_unit * prototypes[query_pred], axis=1)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    logit_train = logit_direction_distance(
        tr.logits, tr.logits, train_pred, train_pred
    )
    logit_eval = logit_direction_distance(
        tr.logits, ev.logits, train_pred, eval_pred
    )

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    train_all_d = g.class_mahalanobis_sq(tr.feats)
    eval_all_d = g.class_mahalanobis_sq(ev.feats)
    feat_train = train_all_d[np.arange(len(tr.labels)), train_pred]
    feat_eval = eval_all_d[np.arange(len(ev.logits)), eval_pred]

    def upper_tail(train_values, values):
        ordered = np.sort(np.asarray(train_values, dtype=float))
        ranks = np.searchsorted(ordered, values, side="right")
        survival = (len(ordered) - ranks + 1.0) / (len(ordered) + 1.0)
        return -np.log(np.maximum(survival, eps))

    logit_tail = upper_tail(logit_train, logit_eval)
    feat_tail = upper_tail(feat_train, feat_eval)
    return logit_tail + feat_tail

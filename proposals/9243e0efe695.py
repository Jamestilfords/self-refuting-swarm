FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A scale-normalized top-two logit gap captures classifier ambiguity, while predicted-intent angular prototype distance captures feature support. Train-fitted empirical tail mappings put both signals on a shared probability scale without using evaluation statistics."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0019", "m_0036"]


def score(ctx):
    import numpy as np

    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    # Logit ambiguity: small top-two separation relative to each vector's
    # intrinsic logit dispersion is more OOS-like.
    def logit_ambiguity(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        gap = top2[:, 1] - top2[:, 0]
        spread = np.std(logits, axis=1) + eps
        return -gap / spread

    tr_logit = logit_ambiguity(tr.logits)
    ev_logit = logit_ambiguity(ev.logits)

    # Label-derived bottleneck prototypes define intent directions.  Use the
    # classifier-predicted intent so feature/logit disagreement is retained.
    xtr = tr.feats / (np.linalg.norm(tr.feats, axis=1, keepdims=True) + eps)
    xev = ev.feats / (np.linalg.norm(ev.feats, axis=1, keepdims=True) + eps)
    n_classes = tr.logits.shape[1]
    prototypes = np.zeros((n_classes, tr.feats.shape[1]), dtype=float)
    counts = np.bincount(tr.labels, minlength=n_classes)

    for c in range(n_classes):
        if counts[c] > 0:
            proto = xtr[tr.labels == c].mean(axis=0)
            prototypes[c] = proto / (np.linalg.norm(proto) + eps)

    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)
    tr_feature = 1.0 - np.sum(xtr * prototypes[tr_pred], axis=1)
    ev_feature = 1.0 - np.sum(xev * prototypes[ev_pred], axis=1)

    # Empirical train CDFs are monotone shared-scale calibrators.  The small
    # endpoint adjustment keeps both components finite and comparable.
    def tail_map(train_scores, test_scores):
        ordered = np.sort(train_scores)
        n = ordered.size
        ranks = np.searchsorted(ordered, test_scores, side="right")
        return (ranks + 0.5) / (n + 1.0)

    logit_tail = tail_map(tr_logit, ev_logit)
    feature_tail = tail_map(tr_feature, ev_feature)

    return 0.5 * logit_tail + 0.5 * feature_tail

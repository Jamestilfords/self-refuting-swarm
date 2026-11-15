import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Known intents should induce similar class competition patterns from classifier logits and "
    "from cosine similarity to L2-normalized feature prototypes. OOS queries can produce angular "
    "prototype evidence that disagrees with the classifier distribution or has weak expected support."
)
SOURCE = "https://github.com/pradhyumnaag30/CLINC150"
PARENTS = []
MEMORY_USED = ["m_0122", "m_0201"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12
    k = tr.logits.shape[1]

    z_tr = tr.feats / np.maximum(
        np.sqrt(np.sum(tr.feats * tr.feats, axis=1, keepdims=True)), eps
    )
    z_ev = ev.feats / np.maximum(
        np.sqrt(np.sum(ev.feats * ev.feats, axis=1, keepdims=True)), eps
    )

    sums = np.zeros((k, z_tr.shape[1]), dtype=float)
    np.add.at(sums, tr.labels, z_tr)
    counts = np.bincount(tr.labels, minlength=k).astype(float)
    proto = sums / np.maximum(counts[:, None], 1.0)
    proto = proto / np.maximum(
        np.sqrt(np.sum(proto * proto, axis=1, keepdims=True)), eps
    )

    sim = z_ev @ proto.T

    logit_prob = ctx.h.softmax(ev.logits)
    angular_logits = 12.0 * sim
    angular_logits = angular_logits - np.max(angular_logits, axis=1, keepdims=True)
    angular_prob = np.exp(angular_logits)
    angular_prob /= np.maximum(np.sum(angular_prob, axis=1, keepdims=True), eps)

    mixture = 0.5 * (logit_prob + angular_prob)
    js = 0.5 * np.sum(
        logit_prob * (
            np.log(np.maximum(logit_prob, eps)) - np.log(np.maximum(mixture, eps))
        ),
        axis=1,
    )
    js += 0.5 * np.sum(
        angular_prob * (
            np.log(np.maximum(angular_prob, eps)) - np.log(np.maximum(mixture, eps))
        ),
        axis=1,
    )

    expected_support = np.sum(logit_prob * sim, axis=1)
    return js + 0.25 * (1.0 - expected_support)

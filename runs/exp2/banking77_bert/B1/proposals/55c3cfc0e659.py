FAMILY = "density"
ARM = "ce"
RATIONALE = "Within-class trailing covariance directions retain low-variance semantic constraints that in-scope queries satisfy, while OOS representations tend to have larger residual energy in those directions."
SOURCE = "https://papers.neurips.cc/paper_files/paper/2022/file/70fa5df8e3300dc30bf19bee44a56155-Paper-Conference.pdf"
PARENTS = []
MEMORY_USED = ["m_0002"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    return partial.bottom(ev.pooled)

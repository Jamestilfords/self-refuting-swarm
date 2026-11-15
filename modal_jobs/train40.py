"""E2 scaffold: run 2 benchmarks x 2 backbones x 2 arms x 5 seeds on Modal.

Write full caches to the sealed volume and only allowed splits to the open
volume. Implementation pending; see design-doc slide 13.

Use TRAIN40_MODAL_OPTIONS on each independent fine-tune function. Four
containers then use at most four B300 GPUs in parallel. The training image
must provide CUDA 13.1 or newer and a compatible PyTorch build.
"""

TRAIN40_MODAL_OPTIONS = {
    "gpu": "B300",
    "cpu": 8,
    "memory": 32 * 1024,  # MiB per fine-tune
    "max_containers": 4,
    "timeout": 2 * 60 * 60,  # seconds per fine-tune
}

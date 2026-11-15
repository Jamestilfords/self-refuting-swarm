"""Shared Modal app and explicitly scoped resources, owned by @XayHanmonty.

Importing this module makes lazy handles only. Provisioning happens in setup.
Job modules import this app and choose their own mounts and secrets.
"""

from pathlib import Path
from uuid import uuid4

import modal

from modal_jobs.config import OPEN_MOUNT, SEALED_MOUNT, Settings


settings = Settings.from_env()
image = (
    modal.Image.debian_slim(python_version="3.12")
    .env(settings.container_env())
    .add_local_python_source("modal_jobs")
)

# No app-wide volumes or secrets: each trusted job explicitly requests its access.
app = modal.App(settings.app_name, image=image)
open_volume = modal.Volume.from_name(
    settings.open_volume_name, environment_name=settings.environment
)
sealed_volume = modal.Volume.from_name(
    settings.sealed_volume_name, environment_name=settings.environment
)
secrets_by_name = {
    name: modal.Secret.from_name(name, environment_name=settings.environment)
    for name in settings.secret_names
}


def _check_volume(role: str, mount: str, forbidden_mount: str, volume: modal.Volume) -> dict:
    """Read/write only a unique setup marker, never dataset files or test scores."""
    root = Path(mount)
    if not root.is_dir():
        raise RuntimeError(f"Expected {role} volume is not mounted.")
    if Path(forbidden_mount).exists():
        raise RuntimeError(f"The {role} check can see the other volume's mount.")

    nonce = uuid4().hex
    directory = root / ".skeptic-setup"
    directory.mkdir(exist_ok=True)
    marker = directory / f"{nonce}.probe"
    try:
        marker.write_text(nonce, encoding="utf-8")
        volume.commit()
        volume.reload()
        if marker.read_text(encoding="utf-8") != nonce:
            raise RuntimeError(f"The {role} volume did not preserve the setup marker.")
    finally:
        # Delete only this invocation's temporary marker, even after a failed check.
        marker.unlink(missing_ok=True)
        volume.commit()

    return {
        "role": role,
        "mount": mount,
        "volume_id": volume.object_id,
        "write_read_commit": "ok",
        "other_mount_absent": True,
    }


@app.function(
    volumes={OPEN_MOUNT: open_volume},
    cpu=0.25,
    memory=128,
    timeout=60,
    max_containers=1,
    scaledown_window=2,
)
def check_open_volume() -> dict:
    return _check_volume("open", OPEN_MOUNT, SEALED_MOUNT, open_volume)


@app.function(
    volumes={SEALED_MOUNT: sealed_volume},
    cpu=0.25,
    memory=128,
    timeout=60,
    max_containers=1,
    scaledown_window=2,
)
def check_sealed_volume() -> dict:
    return _check_volume("sealed", SEALED_MOUNT, OPEN_MOUNT, sealed_volume)

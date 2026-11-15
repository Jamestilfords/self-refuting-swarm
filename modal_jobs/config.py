"""E1 configuration. Values here are resource names, never credential values."""

from dataclasses import dataclass
import os
import re
from typing import Mapping


OPEN_MOUNT = "/open"
SEALED_MOUNT = "/sealed"
_RESOURCE_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


@dataclass(frozen=True)
class Settings:
    app_name: str = "skeptic"
    environment: str = "main"
    open_volume_name: str = "skeptic-open"
    sealed_volume_name: str = "skeptic-sealed"
    secret_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = {
            "app": self.app_name,
            "environment": self.environment,
            "open volume": self.open_volume_name,
            "sealed volume": self.sealed_volume_name,
        }
        for kind, name in names.items():
            if not _RESOURCE_NAME.fullmatch(name):
                raise ValueError(f"Invalid {kind} name: use 1-64 letters, digits, '_' or '-'.")
        for name in self.secret_names:
            if not _RESOURCE_NAME.fullmatch(name):
                raise ValueError("Invalid secret name: use 1-64 letters, digits, '_' or '-'.")
        if self.open_volume_name == self.sealed_volume_name:
            raise ValueError("Open and sealed volumes must have different names.")
        if len(set(self.secret_names)) != len(self.secret_names):
            raise ValueError("Secret names must be unique.")

    def container_env(self) -> dict[str, str]:
        """Carry only resource names into containers that re-import the app."""
        return {
            "SKEPTIC_APP_NAME": self.app_name,
            "MODAL_ENVIRONMENT": self.environment,
            "SKEPTIC_OPEN_VOLUME": self.open_volume_name,
            "SKEPTIC_SEALED_VOLUME": self.sealed_volume_name,
            "SKEPTIC_SECRET_NAMES": ",".join(self.secret_names),
        }

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        values = os.environ if env is None else env
        raw_secrets = values.get("SKEPTIC_SECRET_NAMES", "")
        names = tuple(name.strip() for name in raw_secrets.split(",")) if raw_secrets else ()
        return cls(
            app_name=values.get("SKEPTIC_APP_NAME", "skeptic"),
            environment=values.get("MODAL_ENVIRONMENT", "main"),
            open_volume_name=values.get("SKEPTIC_OPEN_VOLUME", "skeptic-open"),
            sealed_volume_name=values.get("SKEPTIC_SEALED_VOLUME", "skeptic-sealed"),
            secret_names=names,
        )

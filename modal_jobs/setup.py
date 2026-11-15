"""Provision E1 or check an existing deployment: python -m modal_jobs.setup."""

import argparse
import importlib
import json
import sys

import modal

from modal_jobs.config import OPEN_MOUNT, SEALED_MOUNT, Settings


JOB_MODULES = ("train40", "seal", "sandbox", "cache_llama", "score_guard")


def validate_report(report: dict, role: str, volume_id: str) -> None:
    expected = {
        "role": role,
        "mount": OPEN_MOUNT if role == "open" else SEALED_MOUNT,
        "volume_id": volume_id,
        "write_read_commit": "ok",
        "other_mount_absent": True,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"The deployed {role} check does not match this configuration.")


def configure(settings: Settings, *, check_only: bool = False) -> dict:
    """Idempotently prepare resources, deploy the app, and verify both mounts.

    Existing named secrets are referenced and validated, never replaced. Empty
    secret configuration is valid for E1 and public Testbed A model downloads.
    """
    # Validate optional dependencies before creating or deploying anything.
    for name in settings.secret_names:
        secret = modal.Secret.from_name(name, environment_name=settings.environment)
        secret.hydrate()

    volumes = {}
    for role, name in (
        ("open", settings.open_volume_name),
        ("sealed", settings.sealed_volume_name),
    ):
        volume = modal.Volume.from_name(
            name,
            environment_name=settings.environment,
            create_if_missing=not check_only,
            version=2,
        )
        volume.hydrate()
        volumes[role] = volume
    if volumes["open"].object_id == volumes["sealed"].object_id:
        raise RuntimeError("Open and sealed resources resolved to the same Modal Volume.")

    if not check_only:
        from modal_jobs.app import app, settings as app_settings

        if settings != app_settings:
            raise RuntimeError("App configuration changed after import; start a new setup process.")
        # Future E2/E3/E4/E6 functions register on the shared app. Include all
        # explicit job modules so a subsequent setup preserves those functions.
        for module in JOB_MODULES:
            importlib.import_module(f"modal_jobs.{module}")
        app.deploy(environment_name=settings.environment)

    deployed = modal.App.lookup(settings.app_name, environment_name=settings.environment)
    checks = {}
    for role, volume in volumes.items():
        function = modal.Function.from_name(
            settings.app_name,
            f"check_{role}_volume",
            environment_name=settings.environment,
        )
        report = function.remote()
        validate_report(report, role, volume.object_id)
        checks[role] = report

    return {
        "status": "ok",
        "app": settings.app_name,
        "app_id": deployed.app_id,
        "dashboard_url": deployed.get_dashboard_url(),
        "environment": settings.environment,
        "volumes": {
            "open": settings.open_volume_name,
            "sealed": settings.sealed_volume_name,
        },
        "validated_secret_names": list(settings.secret_names),
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only", action="store_true",
        help="Verify the existing app and volumes without creating or redeploying resources.",
    )
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_env()
        with modal.enable_output():
            result = configure(settings, check_only=args.check_only)
    except modal.exception.AuthError:
        print("Modal authentication failed. Run `uv run modal token new` and retry.", file=sys.stderr)
        return 1
    except modal.exception.NotFoundError:
        print(
            "A configured Modal resource was not found. Check MODAL_ENVIRONMENT and "
            "SKEPTIC_SECRET_NAMES; create named secrets in Modal first. "
            "For an initial deployment, run `make setup`.",
            file=sys.stderr,
        )
        return 1
    except (ValueError, RuntimeError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

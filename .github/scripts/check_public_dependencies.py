"""Keep the committed dependency lockfile installable from public PyPI."""

from __future__ import annotations

import tomllib
from pathlib import Path
from urllib.parse import urlsplit


def is_public_url(url: str, host: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme == "https" and parsed.netloc == host


def violations(lock: dict) -> list[str]:
    errors = []
    for package in lock["package"]:
        name = package["name"]
        source = package["source"]
        if source == {"editable": "."}:
            continue
        registry = source.get("registry", "")
        if registry.rstrip("/") != "https://pypi.org/simple":
            errors.append(f"{name}: dependency source must be public PyPI")
        artifacts = [*package.get("wheels", [])]
        if "sdist" in package:
            artifacts.append(package["sdist"])
        for artifact in artifacts:
            if not is_public_url(artifact.get("url", ""), "files.pythonhosted.org"):
                errors.append(
                    f"{name}: distribution URL must use https://files.pythonhosted.org"
                )
    return errors


def main() -> None:
    errors = violations(tomllib.loads(Path("uv.lock").read_text()))
    if errors:
        print("uv.lock contains non-public dependency sources:")
        print("\n".join(errors))
        print(
            "Regenerate with: uv lock --no-config --default-index https://pypi.org/simple"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()

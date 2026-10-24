"""Reject Modal endpoint definitions that bypass proxy authentication."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

WEB_DECORATORS = {
    "fastapi_endpoint",
    "web_endpoint",
    "web_server",
    "asgi_app",
    "wsgi_app",
}


def violations(source: str) -> list[tuple[int, str]]:
    tree = ast.parse(source)
    aliases = {
        alias.asname or alias.name: alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("modal")
        for alias in node.names
    }
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "unauthenticated":
            if (
                not isinstance(node.value, ast.Constant)
                or node.value.value is not False
            ):
                errors.append(
                    (node.lineno, "unauthenticated must be omitted or literal False")
                )
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "unauthenticated":
                    if not isinstance(value, ast.Constant) or value.value is not False:
                        errors.append(
                            (key.lineno, "unauthenticated must be literal False")
                        )
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for decorator in node.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            func = call.func if call else decorator
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            if aliases.get(name, name) not in WEB_DECORATORS:
                continue
            auth = (
                next(
                    (
                        kw.value
                        for kw in call.keywords
                        if kw.arg == "requires_proxy_auth"
                    ),
                    None,
                )
                if call
                else None
            )
            if not isinstance(auth, ast.Constant) or auth.value is not True:
                errors.append(
                    (
                        decorator.lineno,
                        "web decorators require literal requires_proxy_auth=True",
                    )
                )
    return errors


def main() -> None:
    paths = (
        subprocess.check_output(["git", "ls-files", "-z", "--", "*.py"])
        .decode()
        .split("\0")
    )
    failed = False
    for name in filter(None, paths):
        path = Path(name)
        if not path.is_file():
            continue
        for line, message in violations(path.read_text()):
            print(f"{path}:{line}: {message}")
            failed = True
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

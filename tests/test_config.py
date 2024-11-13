"""Tests for the parse_args function."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import msgspec
import pytest

from caldav2ics import Config, parse_args

if TYPE_CHECKING:
    from pathlib import Path


class Patcher:
    def __init__(self, func: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patcher."""
        self.func = func
        self.called = False
        monkeypatch.setattr(self.func, self._return_input_assert_called)

    def _return_input_assert_called(self, *unused_args: Any) -> str:
        self.called = True
        return f"password_{self.func}"

    def __enter__(self) -> None:
        return None

    def __exit__(self, *unused_args: object) -> None:
        assert self.called


@pytest.fixture
def patcher(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Patcher:
    """Patch getpass."""
    return Patcher(request.param, monkeypatch)


@pytest.fixture
def configs() -> dict[str, Config]:
    """Configs."""
    return {
        "caldav": {
            "url": "https://caldav.example.com",
            "user": "user",
            "save-dir": "/tmp/calendars",  # noqa: S108
        }
    }


@pytest.mark.parametrize(
    ("funcname", "args"), [("getpass.getpass", []), ("sys.stdin.read", ["-s"])]
)
def test_parse_args(
    funcname: str,
    args: list[str],
    configs: dict[str, Config],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test parse_args."""
    monkeypatch.setattr(funcname, lambda *_: "password")
    (tmp_path / "config.toml").write_bytes(msgspec.toml.encode(configs))

    args = [str(tmp_path / "config.toml"), *args]
    parsed_configs, passwd = parse_args(args)

    assert parsed_configs == configs
    assert passwd == "password"  # noqa: S105

"""CDP supervisors are per served profile on a multiplexed gateway (#110032, #134889).

The registry was keyed by task id alone, so two profiles whose browser work used the same task id
(``browser_exec`` attaches as ``task_id or "default"``; cron and kanban ids repeat too) shared one
entry: the second profile's ``get_or_start`` saw another ``cdp_url``, stopped the first profile's
supervisor and replaced it, and ``browser_vault`` fills / dialog handling for the first profile then
ran against the other profile's browser or none. Outside a served-profile scope keys are unchanged.
"""

from __future__ import annotations

import contextlib
import threading

import pytest

from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override
from tools import browser_supervisor as bs

_CDPSupervisor = bs.CDPSupervisor


class _StubSupervisor:
    def __init__(self, *, task_id, cdp_url, dialog_policy, dialog_timeout_s):
        self.task_id, self.cdp_url = task_id, cdp_url
        self._release = threading.Event()
        self._thread = threading.Thread(target=self._release.wait, daemon=True)
        self._thread.start()
        self._loop = type("Loop", (), {"is_running": lambda _self: True})()
        self.stopped = False

    def start(self, timeout: float = 15.0) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True
        self._release.set()


@pytest.fixture
def registry(monkeypatch):
    registry = bs._SupervisorRegistry()
    monkeypatch.setattr(bs, "CDPSupervisor", _StubSupervisor)
    monkeypatch.setattr(bs, "SUPERVISOR_REGISTRY", registry)
    yield registry
    registry.stop_all()


@pytest.fixture
def homes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    return a, b


@contextlib.contextmanager
def _served(home):
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def test_two_profiles_with_the_same_task_id_keep_their_own_supervisors(registry, homes):
    a, b = homes
    with _served(a):
        sup_a = registry.get_or_start("default", "http://127.0.0.1:9301")
    with _served(b):
        sup_b = registry.get_or_start("default", "http://127.0.0.1:9302")

    assert sup_a is not sup_b and not sup_a.stopped
    with _served(a):
        assert registry.get("default") is sup_a
    with _served(b):
        assert registry.get("default") is sup_b
        registry.stop("default")
    assert sup_b.stopped and not sup_a.stopped
    with _served(a):
        assert registry.get("default") is sup_a


def test_reaping_one_profile_keeps_the_other_profiles_supervisor_for_the_same_task(registry, homes):
    a, b = homes
    with _served(a):
        sup_a = registry.get_or_start("t", "http://127.0.0.1:9301")
    with _served(b):
        sup_b = registry.get_or_start("t", "http://127.0.0.1:9302")

    assert not sup_a.stopped
    registry.stop_home(hermes_home_key(a))

    assert sup_a.stopped and not sup_b.stopped
    with _served(b):
        assert registry.get("t") is sup_b


def test_a_supervisor_that_gives_up_drops_its_own_entry_from_its_unscoped_thread(registry, homes):
    a, b = homes
    with _served(a):
        sup_a = registry.get_or_start("t", "http://127.0.0.1:9301")
    with _served(b):
        sup_b = registry.get_or_start("t", "http://127.0.0.1:9302")

    # The supervisor's own thread carries no profile scope.
    assert _CDPSupervisor._reconnect_budget_spent(sup_a, bs.MAX_POST_ATTACH_RECONNECT_FAILURES, OSError("gone"))

    with _served(a):
        assert registry.get("t") is None
    with _served(b):
        assert registry.get("t") is sup_b


def test_keys_outside_a_served_profile_are_unchanged(registry):
    sup = registry.get_or_start("t", "http://127.0.0.1:9301")

    assert registry._by_task == {"t": sup}
    assert registry.get("t") is sup

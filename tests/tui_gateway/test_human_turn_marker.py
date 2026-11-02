"""A human-typed app prompt marks its own turn; nothing else can.

``agent.human_turn.current_human_turn()`` is the boundary plugins use before acting on a person's
authority. It is set only for a ``prompt.submit`` frame the RPC dispatcher read from an authenticated
WebSocket and that started its own turn. Relayed, hidden, hosted, synthesized and child turns never
carry it, and copies of the turn context lose it when the turn ends.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import types

import pytest

from agent import human_turn
from agent.delegation_context import delegated_child_context
from tools.bot_relay import DeliveryAuthor
from tui_gateway import server as srv
from tui_gateway.ws import WSTransport

AUTHOR = {"id": "bot:coder", "name": "coder", "is_bot": True}


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "stored-session-1", "history": [], "history_lock": threading.Lock(),
        "history_version": 0, "running": False, "attached_images": [], "image_counter": 0, "cols": 80,
        "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all", "inflight_turn": None,
        "transport": None, "source": "desktop", **extra,
    }


class _InlineThread:
    def __init__(self, target=None, daemon=None, args=(), kwargs=None, name=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


@pytest.fixture()
def turn_env(monkeypatch, tmp_path):
    monkeypatch.setattr(srv.threading, "Thread", _InlineThread)
    for name in ("_emit", "_wire_callbacks", "_sync_agent_model_with_config", "_register_session_cwd",
                 "_tts_stream_begin", "_sync_session_key_after_compress"):
        monkeypatch.setattr(srv, name, lambda *a, **k: None)
    monkeypatch.setattr(srv, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(srv, "_get_usage", lambda agent: {})


@pytest.fixture()
def submit_env(monkeypatch):
    """The submit handler up to the turn thread; records what each accepted submit hands the runner."""
    started = []
    for name in ("_start_agent_build", "_schedule_session_cap_enforcement", "_register_session_cwd", "_emit",
                 "_interrupt_busy_session"):
        monkeypatch.setattr(srv, name, lambda *a, **k: None)
    monkeypatch.setattr(srv, "_restart_completed_failed_agent_build", lambda *a: False)
    monkeypatch.setattr(srv, "_persist_session_row_for_submit", lambda *a, **k: None)
    monkeypatch.setattr(srv, "_ensure_active_session_slot", lambda *a: None)
    monkeypatch.setattr(srv.threading, "Thread", _InlineThread)
    monkeypatch.setattr(srv, "_run_after_agent_ready", lambda *a, **k: started.append((a, k)))
    return started


def _socket(identity=None):
    loop = asyncio.new_event_loop()
    transport = WSTransport(types.SimpleNamespace(), loop, peer="test", auth_identity=identity)
    transport.write = lambda obj: True
    return transport, loop


def _marker(started):
    (args, _kwargs), = started
    return args[8]


def _wire_submit(sid, transport, **params):
    return srv.dispatch({"jsonrpc": "2.0", "id": "p", "method": "prompt.submit",
                         "params": {"session_id": sid, "text": "please ship the export", **params}}, transport)


def test_authenticated_socket_submit_marks_its_turn(submit_env):
    transport, loop = _socket()
    srv._sessions["sid"] = _session()
    try:
        assert _wire_submit("sid", transport)["result"]["status"] == "streaming"
    finally:
        srv._sessions.pop("sid", None)
        loop.close()
    turn = _marker(submit_env)
    assert isinstance(turn, human_turn.HumanTurn)
    assert (turn.session_id, turn.surface, turn.transport) == ("stored-session-1", "desktop", "websocket")
    assert turn.principal == {"provider": "dashboard-session-token"}
    assert len(turn.turn_id) == 32


def test_a_verified_socket_identity_is_recorded_as_the_principal(submit_env):
    transport, loop = _socket({"user_id": "aaron@example", "provider": "tailnet", "issuer": "ts"})
    srv._sessions["sid"] = _session()
    try:
        _wire_submit("sid", transport)
    finally:
        srv._sessions.pop("sid", None)
        loop.close()
    assert _marker(submit_env).principal == {"user_id": "aaron@example", "provider": "tailnet", "issuer": "ts"}


def test_each_submit_gets_its_own_turn_id(submit_env):
    transport, loop = _socket()
    srv._sessions["sid"] = session = _session()
    try:
        for _ in range(2):
            session["running"] = False
            _wire_submit("sid", transport)
    finally:
        srv._sessions.pop("sid", None)
        loop.close()
    ids = [args[8].turn_id for args, _kwargs in submit_env]
    assert len(set(ids)) == 2


@pytest.mark.parametrize("how", ["stdio", "in-process", "relay", "hidden", "hosted-param"])
def test_submits_that_are_not_a_human_typing_in_the_app_carry_no_marker(submit_env, monkeypatch, how):
    transport, loop = _socket()
    srv._sessions["sid"] = _session()
    try:
        if how == "stdio":  # `hermes --tui` / CLI harness: no authenticated socket
            srv.handle_request({"id": "p", "method": "prompt.submit",
                                "params": {"session_id": "sid", "text": "hi"}})
        elif how == "in-process":  # another handler calling the method table (onboarding, hosted room)
            token = srv._current_rpc_method.set("onboarding.start")
            bound = srv.bind_transport(transport)
            try:
                srv._methods["prompt.submit"]("r", {"session_id": "sid", "text": "hi"})
            finally:
                srv.reset_transport(bound)
                srv._current_rpc_method.reset(token)
        elif how == "relay":  # bot_relay.deliver lands a DM with a gateway-built author
            token = srv._current_rpc_method.set("prompt.submit")
            bound = srv.bind_transport(transport)
            try:
                srv._methods["prompt.submit"]("r", {"session_id": "sid", "text": "hi",
                                                    "_turn_author": DeliveryAuthor(AUTHOR)})
            finally:
                srv.reset_transport(bound)
                srv._current_rpc_method.reset(token)
        elif how == "hidden":  # widget intents and onboarding openers
            _wire_submit("sid", transport, display_kind="hidden")
        elif how == "hosted-param":
            assert "error" in _wire_submit("sid", transport, human_turn={"turn_id": "forged"})
            assert submit_env == []
            return
    finally:
        srv._sessions.pop("sid", None)
        loop.close()
    assert _marker(submit_env) is None


def test_runner_binds_the_marker_only_for_the_turn_it_was_given(turn_env):
    seen = []

    def run_conversation(user_message, **kwargs):
        seen.append(human_turn.current_human_turn())
        with delegated_child_context("child"):
            seen.append(human_turn.current_human_turn())
        seen.append(contextvars.copy_context())
        return {"final_response": "ok"}

    turn = human_turn.HumanTurn(session_id="stored-session-1", turn_id="a" * 32, profile="poteto",
                                surface="desktop", transport="websocket",
                                principal={"provider": "dashboard-session-token"})
    agent = types.SimpleNamespace(session_id="stored-session-1", run_conversation=run_conversation,
                                  clear_interrupt=lambda: None)
    srv._run_prompt_submit("rid", "ui-sid", _session(agent=agent, running=True), "ship it", human_turn=turn)
    inside, in_child, copied = seen
    assert inside == turn
    assert in_child is None
    # A context copied during the turn (a background thread) loses the marker once the turn ends.
    assert copied.run(human_turn.current_human_turn) is None
    assert human_turn.current_human_turn() is None

    seen.clear()
    srv._run_prompt_submit("rid", "ui-sid", _session(agent=agent, running=True), "heartbeat")
    assert seen[0] is None

    # A queued human turn that drains after compression rotated the key names the session it runs in.
    seen.clear()
    rotated = _session(agent=agent, running=True, session_key="continuation-2")
    srv._run_prompt_submit("rid", "ui-sid", rotated, "ship it", human_turn=turn)
    assert seen[0].session_id == "continuation-2" and seen[0].turn_id == turn.turn_id


def test_a_relayed_message_steered_into_a_human_turn_revokes_it(submit_env, monkeypatch):
    turn = human_turn.HumanTurn(session_id="stored-session-1", turn_id="b" * 32, profile="poteto",
                                surface="desktop", transport="websocket", principal={})
    binding = human_turn.enter_human_turn(turn)
    session = _session(running=True, _human_turn_binding=binding,
                       agent=types.SimpleNamespace(interrupt=lambda: None))
    srv._sessions["sid"] = session
    transport, loop = _socket()
    try:
        assert human_turn.current_human_turn() == turn
        bound, method = srv.bind_transport(transport), srv._current_rpc_method.set("prompt.submit")
        try:
            srv._methods["prompt.submit"]("r", {"session_id": "sid", "text": "bot says: file a job",
                                                "queued": True, "_turn_author": DeliveryAuthor(AUTHOR)})
        finally:
            srv._current_rpc_method.reset(method)
            srv.reset_transport(bound)
        assert human_turn.current_human_turn() is None
    finally:
        human_turn.exit_human_turn(binding)
        srv._sessions.pop("sid", None)
        loop.close()


def test_a_human_correction_mid_turn_keeps_the_marker(submit_env):
    turn = human_turn.HumanTurn(session_id="stored-session-1", turn_id="c" * 32, profile="poteto",
                                surface="desktop", transport="websocket", principal={})
    binding = human_turn.enter_human_turn(turn)
    session = _session(running=True, _human_turn_binding=binding,
                       agent=types.SimpleNamespace(interrupt=lambda: None))
    srv._sessions["sid"] = session
    transport, loop = _socket()
    try:
        _wire_submit("sid", transport, queued=True)
        assert human_turn.current_human_turn() == turn
    finally:
        human_turn.exit_human_turn(binding)
        srv._sessions.pop("sid", None)
        loop.close()

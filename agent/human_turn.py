"""The human-typed turn boundary for the app's ``prompt.submit`` path.

``prompt.submit`` mints a :class:`HumanTurn` only for a frame the RPC dispatcher read from an
authenticated WebSocket (``tui_gateway.ws``) that is not relayed, hidden or hosted. The turn
runner binds it for exactly that turn. Plugins call :func:`current_human_turn` before acting on
a person's authority; it is None for every other turn: heartbeats, loop wakeups, notifications,
delegation completions, relayed bot messages, cron runs and delegate_task children.

The binding is a revocable cell, not a bare value: a context copied during the turn (a background
thread, a delegation record) sees nothing once the turn ends, and non-human input steered into the
live turn revokes it. JSON clients cannot construct one: the RPC contract forbids unknown keys.
"""
from __future__ import annotations

import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Mapping, Optional


@dataclass(frozen=True)
class HumanTurn:
    session_id: str
    turn_id: str
    profile: str
    surface: str
    transport: str
    principal: Mapping[str, str] = field(default_factory=dict)


class HumanTurnBinding:
    """One turn's live marker. ``revoke`` is idempotent and safe from any thread."""

    __slots__ = ("_live", "_lock", "_token", "turn")

    def __init__(self, turn: HumanTurn) -> None:
        self.turn, self._lock, self._live, self._token = turn, threading.Lock(), True, None

    @property
    def live(self) -> bool:
        with self._lock:
            return self._live

    def revoke(self) -> None:
        with self._lock:
            self._live = False


_CURRENT: ContextVar[Optional[HumanTurnBinding]] = ContextVar("hermes_human_turn", default=None)


def enter_human_turn(turn: HumanTurn) -> HumanTurnBinding:
    """Bind *turn* in the current context; pair with :func:`exit_human_turn` in a ``finally``."""
    if not isinstance(turn, HumanTurn):
        raise TypeError("a human turn is minted by prompt.submit, never supplied")
    binding = HumanTurnBinding(turn)
    binding._token = _CURRENT.set(binding)
    return binding


def exit_human_turn(binding: Optional[HumanTurnBinding]) -> None:
    if binding is None:
        return
    binding.revoke()
    if binding._token is not None:
        _CURRENT.reset(binding._token)
        binding._token = None


def current_human_turn() -> Optional[HumanTurn]:
    """The live human turn this code runs for, else None (children and cron runs never qualify)."""
    binding = _CURRENT.get()
    if binding is None or not binding.live:
        return None
    from agent.delegation_context import is_dispatcher_owned_worker_context
    # False inside a delegate_task child, its descendants and an in-process cron run.
    if not is_dispatcher_owned_worker_context():
        return None
    return binding.turn

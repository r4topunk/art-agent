from collections import defaultdict
from collections.abc import Callable
from typing import Any
import threading


class EventBus:
    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable]] = defaultdict(list)
        self._lock = threading.Lock()

    def on(self, event: str, callback: Callable[..., Any]) -> None:
        with self._lock:
            self._listeners[event].append(callback)

    def off(self, event: str, callback: Callable[..., Any]) -> None:
        with self._lock:
            if callback in self._listeners[event]:
                self._listeners[event].remove(callback)

    def emit(self, event: str, **kwargs: Any) -> None:
        with self._lock:
            listeners = list(self._listeners.get(event, []))
        for cb in listeners:
            cb(**kwargs)

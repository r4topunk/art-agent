"""WebSocket bridge: forwards EventBus events to connected browsers as JSON."""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import numpy as np
import websockets
from websockets.asyncio.server import serve

from art.events import EventBus

EVENTS = [
    "neural_activity", "train_start", "train_step", "train_end",
    "gen_start", "gen_progress", "gen_pieces", "gen_scored",
    "gen_selected", "gen_confidences", "gen_complete",
    "finetune_start", "scoring_start", "scoring_progress",
    "saving_start", "saving_complete", "init_phase",
    "evolution_step",
]


def _serialize(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    return obj


class WebBridge:
    def __init__(self, bus: EventBus, host: str = "0.0.0.0", port: int = 8765):
        self.bus = bus
        self.host = host
        self.port = port
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        self._subscribe_all()
        async with serve(self._handler, self.host, self.port, origins=None):
            print(f"[WebBridge] ws://{self.host}:{self.port}")
            await asyncio.Future()  # run forever

    async def _handler(self, ws):
        self._clients.add(ws)
        try:
            async for _ in ws:
                pass  # we don't expect messages from the client
        finally:
            self._clients.discard(ws)

    def _subscribe_all(self) -> None:
        for event in EVENTS:
            self.bus.on(event, lambda _e=event, **kw: self._forward(_e, kw))

    def _forward(self, event: str, data: dict) -> None:
        if not self._clients or self._loop is None:
            return
        msg = json.dumps({"event": event, "data": _serialize(data)})
        asyncio.run_coroutine_threadsafe(self._broadcast(msg), self._loop)

    async def _broadcast(self, msg: str) -> None:
        dead = set()
        for ws in self._clients:
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        self._clients -= dead

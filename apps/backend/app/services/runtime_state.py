from __future__ import annotations

from typing import Optional


class RuntimeState:
    def __init__(self) -> None:
        self.active_config: Optional[dict] = None


runtime_state = RuntimeState()

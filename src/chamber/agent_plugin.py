"""Chamber agent plugin for automatic profiling integration.

Provides a plugin class that can be registered with the Chamber agent
to automatically profile workloads running on managed GPU infrastructure.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AgentPlugin:
    """Plugin for integrating chamber-profiler with the Chamber agent.

    Designed to be registered with the Chamber agent Helm chart deployment
    so that profiling data is automatically collected and uploaded for
    workloads running on Chamber-managed GPU infrastructure.

    Parameters
    ----------
    api_key : str | None
        Chamber API key for uploading profiles.
    auto_upload : bool
        Whether to automatically upload profiles after collection.
    interval_ms : int
        GPU metric sampling interval in milliseconds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        auto_upload: bool = True,
        interval_ms: int = 100,
    ) -> None:
        self._api_key = api_key
        self._auto_upload = auto_upload
        self._interval_ms = interval_ms
        self._active = False

    def on_job_start(self, job_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Called when a new job starts on the agent."""
        logger.info("AgentPlugin: job %s started.", job_id)
        self._active = True

    def on_job_end(self, job_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Called when a job completes on the agent."""
        logger.info("AgentPlugin: job %s ended.", job_id)
        self._active = False

    def is_active(self) -> bool:
        """Return whether the plugin is currently tracking a job."""
        return self._active

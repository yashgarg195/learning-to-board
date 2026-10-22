"""In-process background training jobs.

DECISION (not a spec requirement): training runs in a daemon thread inside the
API process and the registry is in memory.  The spec forbids introducing a
broker or database, and a tabular policy-gradient run is seconds-to-minutes on
one core, so a thread plus a dict is the simplest correct thing.  The
limitation is that jobs do not survive a restart -- trained policies do,
because they are written to ``artifacts/policies``.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple

MAX_JOBS = 32


class Job:
    def __init__(self, episodes_total: int):
        self.job_id = uuid.uuid4().hex[:12]
        self.status = "queued"
        self.created_at = time.time()
        self.episodes_done = 0
        self.episodes_total = episodes_total
        self.message: Optional[str] = None
        self.curve: List[List[float]] = []
        self.result: Optional[Dict] = None
        self.lock = threading.Lock()

    def snapshot(self) -> Dict:
        with self.lock:
            return {
                "job_id": self.job_id,
                "status": self.status,
                "created_at": self.created_at,
                "episodes_done": self.episodes_done,
                "episodes_total": self.episodes_total,
                "message": self.message,
                "curve": [list(p) for p in self.curve],
                "result": self.result,
            }

    def progress(self, done: int, total: int,
                 curve: List[Tuple[int, float]]) -> None:
        with self.lock:
            self.episodes_done = done
            self.episodes_total = total
            self.curve = [[float(e), float(v)] for e, v in curve]


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()

    def create(self, episodes_total: int) -> Job:
        job = Job(episodes_total)
        with self._lock:
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            while len(self._order) > MAX_JOBS:
                self._jobs.pop(self._order.pop(0), None)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[Job]:
        with self._lock:
            return [self._jobs[j] for j in reversed(self._order)
                    if j in self._jobs]

    def run(self, job: Job, fn) -> None:
        def target() -> None:
            with job.lock:
                job.status = "running"
            try:
                result = fn(job)
                with job.lock:
                    job.result = result
                    job.status = "done"
                    job.episodes_done = job.episodes_total
            except Exception as exc:  # surfaced to the UI, not swallowed
                with job.lock:
                    job.status = "error"
                    job.message = "{}: {}".format(type(exc).__name__, exc)

        threading.Thread(target=target, daemon=True).start()


REGISTRY = JobRegistry()

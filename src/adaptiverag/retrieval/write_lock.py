"""One write at a time per collection, across the processes on this machine."""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from types import TracebackType

import structlog

if sys.platform != "win32":
    import fcntl

logger = structlog.get_logger(__name__)

_POLL_SECONDS = 0.1


def default_write_lock_path(collection_name: str) -> Path | None:
    """The lock file that processes run by this user take turns on to write ``collection_name``.

    It sits in the user's cache directory, so the API and the CLI indexer find the same file.
    None on Windows, which has no ``flock``; writes there take turns only within a process.
    """
    if sys.platform == "win32":
        return None
    try:
        cache_home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    except RuntimeError:  # no home directory to put it in
        return None
    file_name = re.sub(r"[^A-Za-z0-9_.-]", "_", collection_name)
    return cache_home / "adaptiverag" / "locks" / f"{file_name}.lock"


class CollectionWriteLock:
    """Lets one write to a collection run at a time.

    Inside a process an asyncio lock queues the writers. With a ``path``, the writer also holds
    an exclusive ``flock`` on that file, so writers in other processes on this machine wait as
    well: a second API worker or the CLI indexer can't upload into the copy of the collection
    that a rebuild is about to replace. The kernel drops the file lock when its process exits,
    so a crashed writer never leaves the collection locked.

    If the lock file can't be opened, writes still go ahead and only take turns within this
    process, as they did before the file lock existed.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._local = asyncio.Lock()
        self._fd: int | None = None

    async def __aenter__(self) -> None:
        await self._local.acquire()
        try:
            self._fd = await self._lock_file()
        except BaseException:
            self._local.release()
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        fd, self._fd = self._fd, None
        try:
            if fd is not None:
                os.close(fd)  # closing the file releases its lock
        finally:
            self._local.release()

    async def _lock_file(self) -> int | None:
        if self.path is None or sys.platform == "win32":
            return None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as exc:
            self._warn_unavailable(exc)
            return None

        waiting = False
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return fd
                except BlockingIOError:
                    if not waiting:
                        waiting = True
                        logger.info(
                            "collection_write_waiting",
                            lock_file=str(self.path),
                            detail="Another process is writing to this collection.",
                        )
                    # Polling, rather than a blocking flock in a thread, lets a cancelled
                    # writer stop waiting without taking the lock later.
                    await asyncio.sleep(_POLL_SECONDS)
        except OSError as exc:  # a file system that doesn't support flock
            os.close(fd)
            self._warn_unavailable(exc)
            return None
        except BaseException:
            os.close(fd)
            raise

    def _warn_unavailable(self, exc: OSError) -> None:
        logger.warning(
            "collection_write_lock_unavailable",
            lock_file=str(self.path),
            error=str(exc),
            detail="Writes take turns only within this process.",
        )

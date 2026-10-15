"""Spill: move oversized content to disk, leave a locator behind.

Spilling is the only reduction that **appends** rather than replaces. That is
what makes it the cheapest: an append does not invalidate the derived-message
cache, so a spilled result costs one event and no rebuild.

The trade is that the content is not gone -- it is *elsewhere*. That is the point.
Compaction discards information and needs a model to summarize it; spilling keeps
every byte and needs only a filename. Whenever a result is too big but still
might matter, spilling is strictly better than summarizing.

Three properties matter for safety:

* **The handle is opaque.** A consumer must not parse it to find the file; it
  passes it back to the store. If the format leaked into callers, changing how
  files are named would break them.
* **The directory is private and writes are exclusive.** Files are created
  ``O_EXCL`` with mode 0600 inside a 0700 root, so a suggested name can never
  redirect a write or overwrite an existing file.
* **Failure is contained.** A spill that cannot be written leaves the original
  content untouched and reports a warning. Losing the ability to offload is an
  inconvenience; losing the content is a data-loss bug.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("arm.context.spill")

#: Directory mode for the spill root: owner-only.
ROOT_MODE = 0o700
#: File mode for spilled content: owner-only.
FILE_MODE = 0o600

#: Default budget: content larger than this is a spill candidate.
DEFAULT_SPILL_THRESHOLD = 24_000

#: How much of the head and tail stay inline, so the model still sees the shape
#: of the result without fetching it.
DEFAULT_INLINE_HEAD = 2000
DEFAULT_INLINE_TAIL = 500


class SpillError(Exception):
    """A spill could not be performed. The caller keeps the original content."""


@dataclass(frozen=True, slots=True)
class Locator:
    """An opaque handle to spilled content.

    The only supported operation is passing it back to the store. It is a
    branded string rather than a path precisely so that no consumer is tempted to
    open it directly -- that temptation is how the on-disk layout becomes a
    public interface by accident.
    """

    token: str

    def __str__(self) -> str:
        return self.token

    def to_record(self) -> dict[str, str]:
        return {"locator": self.token}


def _new_token() -> str:
    """An unguessable token. Not derived from the content or the session id.

    A content-derived name would leak whether two sessions saw the same data, and
    a session-derived one would be predictable. Neither matters for correctness
    and both matter for privacy, so the name is random.
    """
    return secrets.token_hex(16)


class SpillStore:
    """A private directory of spilled content, addressed by opaque tokens."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=ROOT_MODE)
        try:
            os.chmod(self.root, ROOT_MODE)
        except OSError:  # pragma: no cover - non-POSIX
            pass

    def put(self, content: str, *, session_id: str = "") -> Locator:
        """Write ``content`` and return its locator.

        The write is exclusive and the file is owner-only. On any failure the
        exception propagates and no partial file is left behind.
        """
        self._ensure_root()
        token = _new_token()
        path = self.root / f"{token}.txt"

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            fd = os.open(path, flags, FILE_MODE)
        except OSError as exc:
            raise SpillError(f"could not create spill file: {exc}") from exc

        try:
            data = content.encode("utf-8")
            written = os.write(fd, data)
            if written != len(data):
                raise SpillError(
                    f"partial spill write: {written} of {len(data)} bytes"
                )
            os.fsync(fd)
        except BaseException as exc:
            # Remove the partial file: a half-written spill would be worse than
            # no spill, because the locator would look valid.
            try:
                os.close(fd)
            finally:
                try:
                    path.unlink()
                except OSError:
                    pass
            if isinstance(exc, SpillError):
                raise
            raise SpillError(f"could not write spill file: {exc}") from exc
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

        if session_id:
            # Record which session produced it, for the sweep. Written as a
            # sidecar so the content file stays exactly the spilled bytes.
            sidecar = self.root / f"{token}.meta"
            try:
                sidecar.write_text(
                    f"session={session_id}\nbytes={len(content)}\n"
                    f"sha256={hashlib.sha256(content.encode()).hexdigest()}\n",
                    encoding="utf-8",
                )
                os.chmod(sidecar, FILE_MODE)
            except OSError:
                # Metadata is diagnostic only; its absence must not fail a spill.
                log.warning("could not write spill metadata for %s", token)

        return Locator(token=token)

    def get(self, locator: Locator | str) -> str:
        """Read spilled content back."""
        token = locator.token if isinstance(locator, Locator) else str(locator)
        path = self._resolve(token)
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SpillError(f"could not read spilled content: {exc}") from exc

    def _resolve(self, token: str) -> Path:
        """Resolve a token to a path, refusing anything that escapes the root.

        The token is validated as hex before use, so a caller cannot pass a path
        fragment or a traversal sequence. This is the check that makes the
        "opaque handle" claim true rather than aspirational.
        """
        if not token or not all(c in "0123456789abcdef" for c in token):
            raise SpillError(f"invalid locator: {token!r}")
        return self.root / f"{token}.txt"

    def exists(self, locator: Locator | str) -> bool:
        token = locator.token if isinstance(locator, Locator) else str(locator)
        try:
            return self._resolve(token).is_file()
        except SpillError:
            return False

    def delete(self, locator: Locator | str) -> bool:
        token = locator.token if isinstance(locator, Locator) else str(locator)
        try:
            path = self._resolve(token)
        except SpillError:
            return False
        try:
            path.unlink()
            (self.root / f"{token}.meta").unlink(missing_ok=True)
            return True
        except OSError:
            return False

    def total_bytes(self) -> int:
        """Total size of spilled content, for capacity decisions."""
        if not self.root.exists():
            return 0
        total = 0
        for entry in self.root.iterdir():
            # Only regular files are measured. Following a symlink here would let
            # a link to a huge file inflate the number, and the sweep that uses
            # it must never traverse links.
            if entry.is_file() and not entry.is_symlink():
                try:
                    total += entry.stat().st_size
                except OSError:
                    continue
        return total

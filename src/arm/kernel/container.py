"""The plugin container: reversible registration, ordered dispatch.

The container is deliberately small. It does three things:

1. **Service registration** -- a name maps to one instance, and the provider is
   handed a disposer that removes exactly its own registration. This is what
   makes unloading a plugin leave nothing behind.
2. **Event dispatch** -- listeners are called in registration order, and their
   failures are contained. A listener that raises must not be able to break the
   commit that triggered it; the log is already the truth by then.
3. **Disposal** -- tearing down runs disposers in reverse registration order, so
   a plugin can rely on its dependencies outliving it.

Nothing here knows about sessions, tools or models. Keeping the container
ignorant is what lets every other layer be replaced without touching it.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Generic, Iterable, TypeVar

from .errors import ServiceAlreadyProvidedError, ServiceNotProvidedError

log = logging.getLogger("arm.kernel.container")

T = TypeVar("T")

Disposer = Callable[[], None]
Listener = Callable[[Any], None]


class Registration(Generic[T]):
    """A live registration. Iterating it yields the current value."""

    __slots__ = ("_value", "_disposed", "_dispose")

    def __init__(self, value: T, dispose: Disposer) -> None:
        self._value = value
        self._disposed = False
        self._dispose = dispose

    @property
    def value(self) -> T:
        return self._value

    @property
    def disposed(self) -> bool:
        return self._disposed

    def dispose(self) -> None:
        """Remove this registration. Idempotent."""
        if self._disposed:
            return
        self._disposed = True
        self._dispose()


class Container:
    """A service registry with reversible, ordered registration.

    Every ``register_*`` call returns a :class:`Registration`; disposing it
    removes exactly that entry. Registrations made later are disposed earlier.
    """

    __slots__ = ("_services", "_listeners", "_disposers", "_disposed")

    def __init__(self) -> None:
        self._services: dict[str, Registration[Any]] = {}
        self._listeners: dict[str, list[Registration[Listener]]] = {}
        self._disposers: list[Registration[Disposer]] = []
        self._disposed = False

    # -- services --------------------------------------------------------

    def provide(self, name: str, instance: Any) -> Registration[Any]:
        """Register ``instance`` under ``name``.

        Replacing an existing provider is refused rather than allowed: two
        providers for one name means the winner depends on load order, and load
        order is not a design decision.
        """
        self._assert_live()
        existing = self._services.get(name)
        if existing is not None and not existing.disposed:
            raise ServiceAlreadyProvidedError(
                f"service {name!r} is already provided by {existing.value!r}"
            )

        def _dispose() -> None:
            if self._services.get(name) is registration:
                del self._services[name]

        registration: Registration[Any] = Registration(instance, _dispose)
        self._services[name] = registration
        return registration

    def get(self, name: str) -> Any:
        """Return the service registered under ``name``, or raise."""
        registration = self._services.get(name)
        if registration is None or registration.disposed:
            raise ServiceNotProvidedError(f"service {name!r} is not provided")
        return registration.value

    def find(self, name: str) -> Any | None:
        """Return the service registered under ``name``, or ``None``."""
        registration = self._services.get(name)
        if registration is None or registration.disposed:
            return None
        return registration.value

    def has(self, name: str) -> bool:
        return self.find(name) is not None

    # -- events ----------------------------------------------------------

    def listen(self, event: str, listener: Listener) -> Registration[Listener]:
        """Subscribe to ``event``. Listeners fire in registration order."""
        self._assert_live()
        bucket = self._listeners.setdefault(event, [])

        def _dispose() -> None:
            if registration in bucket:
                bucket.remove(registration)

        registration: Registration[Listener] = Registration(listener, _dispose)
        bucket.append(registration)
        return registration

    def emit(self, event: str, payload: Any = None) -> None:
        """Dispatch ``event`` to every listener.

        A raising listener is logged and skipped. Containment is the point: by
        the time events are emitted the durable work has already happened, so a
        broken observer must not be able to unwind it.
        """
        for registration in tuple(self._listeners.get(event, ())):
            if registration.disposed:
                continue
            try:
                registration.value(payload)
            except Exception:  # noqa: BLE001 - containment is deliberate
                log.exception("listener for %r failed; continuing", event)

    # -- lifecycle -------------------------------------------------------

    def on_dispose(self, disposer: Disposer) -> Registration[Disposer]:
        """Register a teardown callback, run in reverse order."""
        self._assert_live()

        def _dispose() -> None:
            if registration in self._disposers:
                self._disposers.remove(registration)

        registration: Registration[Disposer] = Registration(disposer, _dispose)
        self._disposers.append(registration)
        return registration

    def dispose(self) -> None:
        """Run every disposer in reverse registration order.

        Every disposer runs even if earlier ones raise; the failures are
        aggregated so that one broken plugin cannot strand the others.
        """
        if self._disposed:
            return
        self._disposed = True
        failures: list[Exception] = []
        for registration in reversed(self._disposers):
            try:
                registration.value()
            except Exception as exc:  # noqa: BLE001 - keep tearing down
                failures.append(exc)
                log.exception("disposer failed during container teardown")
        self._disposers.clear()
        self._services.clear()
        self._listeners.clear()
        if failures:
            raise ExceptionGroup("container teardown had failures", failures)

    def _assert_live(self) -> None:
        if self._disposed:
            raise RuntimeError("container has been disposed")

    def __enter__(self) -> "Container":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.dispose()


def dispose_all(registrations: Iterable[Registration[Any]]) -> None:
    """Dispose a set of registrations, newest first, aggregating failures."""
    failures: list[Exception] = []
    for registration in reversed(list(registrations)):
        try:
            registration.dispose()
        except Exception as exc:  # noqa: BLE001
            failures.append(exc)
    if failures:
        raise ExceptionGroup("registration teardown had failures", failures)

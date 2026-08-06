import signal
import threading

import pytest

from pykappa import _utils
from pykappa._utils import uninterruptible


@pytest.fixture(autouse=True)
def isolate_sigint():
    """Give each test a known SIGINT baseline and reset the shared guard state."""
    guard = _utils._interrupt_guard
    saved = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    guard._depth = 0
    guard._pending = None
    guard._original = None
    yield
    guard._depth = 0
    guard._pending = None
    signal.signal(signal.SIGINT, saved)


def _spin():
    for _ in range(10000):
        pass


def test_defers_interrupt_until_completion():
    completed = []

    @uninterruptible
    def critical():
        signal.raise_signal(signal.SIGINT)
        _spin()  # the handler runs here and must only defer, not interrupt
        completed.append(True)

    with pytest.raises(KeyboardInterrupt):
        critical()
    assert completed == [True]  # ran to completion before the interrupt surfaced


def test_idle_interrupt_surfaces_immediately():
    @uninterruptible
    def noop():
        pass

    noop()  # installs the persistent handler; guard is now idle
    with pytest.raises(KeyboardInterrupt):
        signal.raise_signal(signal.SIGINT)
        _spin()


def test_nested_calls_defer_to_outermost():
    order = []

    @uninterruptible
    def inner():
        signal.raise_signal(signal.SIGINT)
        _spin()
        order.append("inner")

    @uninterruptible
    def outer():
        inner()
        order.append("outer")  # deferred past inner's return, so this still runs

    with pytest.raises(KeyboardInterrupt):
        outer()
    assert order == ["inner", "outer"]


def test_delegates_to_preexisting_handler():
    calls = []

    def user_handler(signum, frame):
        calls.append(signum)

    signal.signal(signal.SIGINT, user_handler)

    @uninterruptible
    def critical():
        signal.raise_signal(signal.SIGINT)
        _spin()
        assert calls == []  # deferred: the user handler has not fired yet

    critical()
    assert calls == [signal.SIGINT]  # delegated to the original handler on exit


def test_ignored_interrupt_is_dropped():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    @uninterruptible
    def critical():
        signal.raise_signal(signal.SIGINT)
        _spin()

    critical()  # SIG_IGN was in place, so nothing is surfaced


def test_worker_thread_runs_without_installing_a_handler():
    result = []

    @uninterruptible
    def work():
        # Installing a handler off the main thread would raise; the wrapper must
        # skip the guard entirely here.
        result.append(True)

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    assert result == [True]

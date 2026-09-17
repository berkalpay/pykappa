import signal
import threading

import pytest

from pykappa._utils import defer_sigint


@pytest.fixture(autouse=True)
def restore_sigint():
    saved = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, saved)


def _spin():
    for _ in range(10000):
        pass


def test_defers_interrupt_until_completion():
    completed = []
    original = signal.getsignal(signal.SIGINT)

    @defer_sigint
    def critical():
        signal.raise_signal(signal.SIGINT)
        _spin()  # the handler runs here and must only defer, not interrupt
        completed.append(True)

    with pytest.raises(KeyboardInterrupt):
        critical()
    assert completed == [True]
    assert signal.getsignal(signal.SIGINT) is original


def test_nested_calls_defer_to_outermost():
    order = []

    @defer_sigint
    def inner():
        signal.raise_signal(signal.SIGINT)
        _spin()
        order.append("inner")

    @defer_sigint
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

    @defer_sigint
    def critical():
        signal.raise_signal(signal.SIGINT)
        _spin()
        assert calls == []  # deferred: the user handler has not fired yet

    critical()
    assert calls == [signal.SIGINT]
    assert signal.getsignal(signal.SIGINT) is user_handler


def test_worker_thread_runs_without_installing_a_handler():
    result = []

    @defer_sigint
    def work():
        # Installing a handler off the main thread would raise; the wrapper must
        # skip the guard entirely here.
        result.append(True)

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    assert result == [True]

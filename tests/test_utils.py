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
        _spin()  # let the deferred handler run
        completed.append(True)

    with pytest.raises(KeyboardInterrupt):
        critical()
    assert completed == [True]
    assert signal.getsignal(signal.SIGINT) is original


def test_nested_calls_preserve_handler():
    calls = []
    order = []

    def handler(signum, frame):
        calls.append(signum)

    signal.signal(signal.SIGINT, handler)

    @defer_sigint
    def inner():
        signal.raise_signal(signal.SIGINT)
        _spin()
        order.append("inner")

    @defer_sigint
    def outer():
        inner()
        order.append("outer")  # still deferred after inner returns

    outer()
    assert order == ["inner", "outer"]
    assert calls == [signal.SIGINT]
    assert signal.getsignal(signal.SIGINT) is handler


def test_worker_thread_runs_without_installing_a_handler():
    result = []

    @defer_sigint
    def work():
        # signal handlers can only be installed on the main thread
        result.append(True)

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    assert result == [True]

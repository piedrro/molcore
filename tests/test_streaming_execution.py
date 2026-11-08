"""Bounded execution, callback threading, cancellation and ordered delivery."""

import threading
import time
import unittest
from concurrent.futures import CancelledError

from molcore.cpu import CpuExecutor
from molcore.execution import SerialExecutor, ThreadExecutor


def worker(value, reporter):
    reporter.check_cancelled()
    if value == 0:
        time.sleep(0.04)
    reporter.report(1, 1, "computed")
    return value * 2


class StreamingTests(unittest.TestCase):
    def test_zero_total_rejects_a_none_payload(self):
        for executor in (SerialExecutor(), ThreadExecutor(num_cpus=1), CpuExecutor(num_cpus=1)):
            with executor, self.assertRaises(ValueError):
                executor.map(lambda value, reporter: value, [None], total=0)

    def test_thread_workers_are_not_limited_by_windows_process_pool_rules(self):
        with ThreadExecutor(num_cpus=64) as executor:
            self.assertEqual(executor.num_cpus, 64)

    def test_ordered_consumption_does_not_submit_unbounded_jobs(self):
        for executor in (
            SerialExecutor(),
            ThreadExecutor(num_cpus=2, max_pending=3),
            CpuExecutor(num_cpus=2, max_pending=3),
        ):
            with self.subTest(executor=type(executor).__name__), executor:
                consumed, callbacks = ([], [])

                def jobs(consumed=consumed):
                    for value in range(8):
                        consumed.append(value)
                        yield value

                stream = executor.imap(
                    worker,
                    jobs(),
                    total=8,
                    progress=lambda update, callbacks=callbacks: callbacks.append(threading.get_ident()),
                )
                self.assertEqual(next(stream), 0)
                self.assertLessEqual(len(consumed), 3)
                self.assertEqual(list(stream), [2, 4, 6, 8, 10, 12, 14])
                self.assertTrue(callbacks)
                self.assertEqual(set(callbacks), {threading.get_ident()})

    def test_early_close_allows_executor_reuse_and_cancellation_preserves_order(self):
        for factory in (SerialExecutor, lambda: ThreadExecutor(num_cpus=2), lambda: CpuExecutor(num_cpus=2)):
            with factory() as executor:
                stream = executor.imap(worker, [0, 1, 2])
                self.assertEqual(next(stream), 0)
                stream.close()
                self.assertEqual(executor.map(worker, [2]), [4])
                stop = threading.Event()
                stop.set()
                with self.assertRaises(CancelledError):
                    list(executor.imap(worker, [1], cancel_event=stop))


if __name__ == "__main__":
    unittest.main()

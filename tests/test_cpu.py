import os
import threading
import time
import unittest
from concurrent.futures import CancelledError
from unittest.mock import patch

from molcore.cpu import CpuExecutor


def cpu_work(value, reporter):
    reporter.report(1, 2, "Halfway")
    time.sleep(0.03)
    reporter.check_cancelled()
    return value * 2, os.getpid()


def cpu_failure(value, reporter):
    raise ValueError("worker failure")


def cpu_wait(value, reporter):
    for i in range(100):
        reporter.report(i, 100, "Working")
        time.sleep(0.005)
    return value


class CpuTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows process-pool limit")
    def test_windows_worker_limit_is_validated_before_allocating_resources(self):
        with patch("molcore.cpu.os.process_cpu_count", return_value=128, create=True):
            self.assertEqual(CpuExecutor().num_cpus, 61)
        with self.assertRaises(ValueError):
            CpuExecutor(num_cpus=62)

    def test_spawn_ordered_results_and_parent_progress(self):
        updates, callback_pids = [], []

        def progress(update):
            updates.append(update)
            callback_pids.append(os.getpid())

        with CpuExecutor(num_cpus=2) as cpu:
            result = cpu.map(cpu_work, [3, 1, 2], progress=progress)
            self.assertEqual(cpu.num_cpus, 2)
        self.assertEqual([v for v, _ in result], [6, 2, 4])
        self.assertTrue(all(pid != os.getpid() for _, pid in result))
        self.assertEqual(set(callback_pids), {os.getpid()})
        percentages = [u.percent for u in updates]
        self.assertEqual(percentages[0], 0)
        self.assertEqual(percentages[-1], 100)
        self.assertEqual(percentages, sorted(percentages))
        self.assertTrue(any(u.description == "Halfway" for u in updates))

    def test_one_cpu_is_still_a_worker_and_executor_can_be_reused(self):
        with CpuExecutor(num_cpus=1) as cpu:
            self.assertNotEqual(cpu.map(cpu_work, [1])[0][1], os.getpid())
            self.assertEqual(cpu.map(cpu_work, [2])[0][0], 4)

    def test_worker_error_and_cancellation_propagate(self):
        with CpuExecutor(num_cpus=1) as cpu:
            with self.assertRaisesRegex(ValueError, "worker failure"):
                cpu.map(cpu_failure, [1])
            self.assertEqual(cpu.map(cpu_work, [2])[0][0], 4)
        stop = threading.Event()

        def progress(update):
            if update.percent > 0:
                stop.set()

        with CpuExecutor(num_cpus=1) as cpu:
            with self.assertRaises(CancelledError):
                cpu.map(cpu_wait, [1], progress=progress, cancel_event=stop)

    def test_jobs_are_consumed_with_a_bounded_pending_window(self):
        consumed = []
        stop = threading.Event()

        def jobs():
            for index in range(100):
                consumed.append(index)
                yield index

        def progress(update):
            if update.description == "Working":
                stop.set()

        with CpuExecutor(num_cpus=1, max_pending=2) as cpu:
            with self.assertRaises(CancelledError):
                cpu.map(
                    cpu_wait, jobs(), total=100, progress=progress, cancel_event=stop
                )
        self.assertLessEqual(len(consumed), 2)

    def test_configuration_empty_jobs_and_pre_cancelled_jobs(self):
        for count in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                CpuExecutor(num_cpus=count)
        updates = []
        with CpuExecutor(num_cpus=1) as cpu:
            self.assertEqual(cpu.map(cpu_work, [], progress=updates.append), [])
            self.assertEqual(updates[-1].percent, 100)
            stop = threading.Event()
            stop.set()
            with self.assertRaises(CancelledError):
                cpu.map(cpu_work, [1], cancel_event=stop)

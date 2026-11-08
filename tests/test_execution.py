import threading
import tomllib
import unittest
from concurrent.futures import CancelledError
from pathlib import Path

from molcore.backends import BackendRegistry, missing_backend, resolve_backend
from molcore.execution import ExecutionContext


class FakeExecutor:
    def map(self, function, jobs, *, total=None, progress=None, cancel_event=None):
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError()
        return [function(job, None) for job in jobs]


class FakeBackend:
    version = "custom-1"

    def compute(self, payload, reporter):
        return payload * 2


class ExecutionTests(unittest.TestCase):
    def test_borrowed_executor_and_callbacks_are_preserved(self):
        cpu, updates, event = (FakeExecutor(), [], threading.Event())
        context = ExecutionContext(cpu=cpu, progress=updates.append, cancel_event=event)
        options = context.kwargs()
        self.assertIs(options["cpu"], cpu)
        self.assertIs(options["cancel_event"], event)
        self.assertEqual(cpu.map(FakeBackend().compute, [2, 3]), [4, 6])
        self.assertIs(context.resolve().cpu, cpu)

    def test_conflicts_and_invalid_counts(self):
        with self.assertRaises(ValueError):
            ExecutionContext(cpu=FakeExecutor(), num_cpus=2)
        for count in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                ExecutionContext(num_cpus=count)
        with self.assertRaises(ValueError):
            ExecutionContext(num_cpus=2).resolve(num_cpus=3)
        with self.assertRaises(ValueError):
            ExecutionContext(cpu=FakeExecutor()).resolve(num_cpus=3)
        self.assertEqual(ExecutionContext().resolve(num_cpus=2).num_cpus, 2)

    def test_cancellation_before_dispatch(self):
        event = threading.Event()
        event.set()
        context = ExecutionContext(cancel_event=event)
        with self.assertRaises(CancelledError):
            context.check_cancelled()
        with self.assertRaises(CancelledError):
            context.kwargs()

    def test_registry_injection_duplicates_and_isolation(self):
        registry, other = (BackendRegistry(), BackendRegistry())
        backend = FakeBackend()
        registry.register("track", "custom", backend)
        self.assertIs(resolve_backend("track", "custom", registry=registry).backend, backend)
        self.assertIs(resolve_backend("track", backend).backend, backend)
        self.assertIsNone(resolve_backend("track", None))
        with self.assertRaises(ValueError):
            registry.register("track", "custom", backend)
        with self.assertRaisesRegex(LookupError, "register"):
            other.resolve("track", "custom")
        with self.assertRaisesRegex(LookupError, "register"):
            registry.resolve("segment", "custom")
        registry.register("track", "custom", backend, replace=True)
        with self.assertRaises(TypeError):
            registry.register("track", "invalid", object())

    def test_missing_backend_has_install_command(self):
        self.assertIn("pip install 'molcore[tracking]'", str(missing_backend("trackpy")))

    def test_processing_dependencies_are_optional(self):
        config = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
        project = config["project"]
        for dependency in ("cellpose", "picassosr", "trackpy", "scikit-learn"):
            self.assertFalse(any(item.startswith(dependency) for item in project["dependencies"]))
            self.assertTrue(
                any(item.startswith(dependency) for item in project["optional-dependencies"]["all"])
            )
        self.assertIn("scipy", project["dependencies"])

    def test_boundary_imports_do_not_load_optional_backends(self):
        import subprocess
        import sys

        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.modules.update({name: None for name in "
                "('cellpose', 'picasso', 'trackpy', 'sklearn')}); "
                "import molcore.backends; import molcore.execution",
            ],
            check=True,
        )

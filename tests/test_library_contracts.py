"""Library documentation, annotation and diagnostic logging contracts."""

import ast
import logging
import unittest
from pathlib import Path

import numpy as np

from molcore import Acquisition, ArraySource, Dataset
from molcore.filtering import filter_results


class LibraryContractTests(unittest.TestCase):
    def test_all_functions_have_parameter_and_return_annotations(self):
        missing = []
        root = Path(__file__).resolve().parents[1] / "src" / "molcore"
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                args += [a for a in (node.args.vararg, node.args.kwarg) if a is not None]
                if node.returns is None or any(
                    a.annotation is None for a in args if a.arg not in ("self", "cls")
                ):
                    missing.append(f"{path.name}:{node.lineno} {node.name}")
        self.assertEqual(missing, [])

    def test_public_functions_document_arguments_and_returns(self):
        missing = []
        root = Path(__file__).resolve().parents[1] / "src" / "molcore"
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                    continue
                doc = ast.get_docstring(node) or ""
                args = [a for a in node.args.args + node.args.kwonlyargs if a.arg not in ("self", "cls")]
                if (
                    not doc
                    or (args and "Args:" not in doc)
                    or (
                        node.returns
                        and ast.unparse(node.returns) != "None"
                        and "Returns:" not in doc
                        and "Yields:" not in doc
                    )
                ):
                    missing.append(f"{path.name}:{node.lineno} {node.name}")
        self.assertEqual(missing, [])

    def test_operation_logs_and_exception_tracebacks_without_array_dump(self):
        model = Dataset()
        model.add_acquisition(Acquisition(ArraySource(np.zeros((1, 3, 3))), id="a"))
        with self.assertLogs("molcore", logging.DEBUG) as captured:
            filter_results(model, "a", "image")
            with self.assertRaises(KeyError):
                filter_results(model, "missing", "image")
        records = captured.records
        self.assertTrue(any("completed" in r.getMessage().lower() for r in records))
        self.assertTrue(any(r.exc_info is not None for r in records))
        self.assertFalse(any("array([" in r.getMessage() for r in records))


if __name__ == "__main__":
    unittest.main()

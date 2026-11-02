from pathlib import Path

p = Path("src/molcore/_locking.py")
s = (
    p.read_text()
    .replace("from typing import ParamSpec, TypeVar\n", "")
    .replace('P = ParamSpec("P")\nR = TypeVar("R")\n\n\n', "")
)
s = s.replace("def model_locked(", "def model_locked[**P, R](")
p.write_text(s)
p = Path("src/molcore/staging.py")
s = (
    p.read_text()
    .replace("from typing import TypeVar\n", "")
    .replace('R = TypeVar("R")\n\n\n', "")
    .replace("def transact(", "def transact[R](")
)
p.write_text(s)
p = Path("docs/scientific-api.md")
s = p.read_text().replace(
    "Use model edit methods in operations.\nDetached processor results",
    """Use model edit methods in operations.
Batch operations may apply detached results sharing the original batch token while
their scientific inputs remain unchanged. Independent observation/provenance updates
can accumulate; edits to localizations, memberships, transforms, acquisition bindings,
masks or experiment metadata invalidate subsequent previews based on the earlier state.
Apply independent photometry results before replacing tracks; dependent processing
must be recomputed from the updated model.
Transactions recheck freshness immediately before adoption. Public model mutators
and adoption share a reentrant lock; direct dictionary or borrowed-array edits are
not synchronized and must not run concurrently with processing.
Detached processor results""",
)
p.write_text(s)
p = Path("README.md")
s = p.read_text().replace(
    "Tracking algorithms remain in the consuming applications.",
    "Shared Trackpy linking, aperture photometry, multichannel traces and trajectory statistics\nare available as headless processors; presentation remains in the consuming applications.",
)
s = s.replace(
    "processing histories and tracking algorithms are outside this release.",
    "full processing histories are outside this release.",
)
p.write_text(s)

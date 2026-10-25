# Typing, documentation and diagnostic logging

Function arguments and returns are annotated throughout molcore, including worker
callbacks and Qt tools. Google-style docstrings describe `Args`, `Returns`, `Yields`
and `Raises` where applicable. Dynamic backend payloads and arbitrary metadata use
`Any` where their contents are deliberately open. Headless imports do not need Qt.

Molcore uses Python's `molcore.*` logger hierarchy with a package `NullHandler`.
It does not call `basicConfig`, install file handlers or change host logging levels.
Configure diagnostics in the embedding application:

```python
import logging

handler = logging.FileHandler("molcore-debug.log", encoding="utf-8")
handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s [%(processName)s/%(threadName)s] %(message)s"
))
logger = logging.getLogger("molcore")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
```

INFO covers scientific processing, imports, rendering and snapshot I/O. DEBUG
covers queries and edits, settings, selection context, progress, job coordination,
transactions and elapsed times. Failures include DEBUG tracebacks before propagating;
Qt job failures also reach `UiSession.error`. Cancellation has a distinct diagnostic
message and preserves active results.

Payload diagnostics show types, array shapes/dtypes, field names and collection
counts. Settings dataclasses include bounded field summaries. Image arrays and
entire measurement tables are not dumped.

Spawned workers do not inherit arbitrary logging handlers. The caller logs job
submission, structured progress, completion and propagated worker failures. Hosts
needing worker-internal diagnostics should configure their worker logging explicitly;
molcore does not install multiprocessing file handlers.

For focused diagnostics, enable DEBUG on `molcore.ui.session` or `molcore.tracking`
while leaving other modules at INFO. The standalone example accepts `--debug` and
`--smoke`, which checks every tab embeds and renders offscreen.

Contract tests check annotation coverage, public Args/Returns sections and traceback
logging. They complement numerical and widget tests; downstream static type checking
remains useful for application-specific schemas.

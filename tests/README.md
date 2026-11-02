# Foundation verification

Run `python -m unittest discover -s tests -v` with MolCore installed. Install the
standard runtime dependencies to exercise every case. The suite covers:

- Original-camera mapping for ALEX, odd tails, logical ROIs and explicit sequences.
- Stable identifiers, detached queries, atomic edits, partition replacement,
  observation invalidation and empty-result schemas.
- Affine/drift parity between points and images, masks and raw image crops.
- Every snapshot image policy in HDF5 and directory layouts, relocation,
  missing images, failed replacement and same-path crop rewrites.
- Picasso metadata, extra measurements, napari axis order, bounded caching,
  Dask evaluation, FITS sections and spawned-process TIFF/HDF5 reads.
- Package import with GUI, analysis engines and optional dependencies blocked.
- Spawn CPU execution with configurable counts, partial progress, cancellation,
  exception propagation, bounded jobs, reuse and Windows worker limits.
- Real Picasso detection, Gaussian LQ/MLE, RCC/AIM drift and multi-acquisition
  undrifting on synthetic data; model-based affine alignment and preservation of
  active results after stale inputs, cancellation or optimizer failure.
- Cellpose 3 constructor/eval compatibility, real preprocessing/mask construction
  with controlled network outputs, real spawned invalid-checkpoint failure,
  inference atomicity, previews, source/target staleness and segmentation persistence.

Verification targets are Python 3.12 and 3.14. The workflow also exercises Windows
and Linux. `examples/foundation.py` demonstrates both application use cases.

The implementation received an independent read-only review. Six reported
integrity issues were reproduced in failing tests and corrected before delivery.
The subsequent processing increment received a separate review; optimizer-failure
handling, Windows CPU limits and RCC segment validation were corrected through
regression tests. Cellpose verification deliberately avoids pretrained-weight
downloads and does not establish pretrained accuracy or GPU performance.

A local synthetic benchmark with 1,000,000 localization rows measured warmed
queries of 0.128 ms for one ID and 0.152 ms for a 1,000-row frame selection.
Index construction is a one-time cost after table edits; these measurements
are host-specific and are not test thresholds.

Scientific API verification adds 34 tests covering measurement identity and
persistence, transactional rollback and concurrency checks, safe dependency cleanup,
partition isolation, affine/ALEX conversions, synthetic photometry, drift, masking,
missing pixels, cancellation, spawned Trackpy workers, actual-time MSD and diffusion.
The new tests and scientific example pass on Python 3.12 and 3.14. The independent
review findings were reproduced and corrected through regression tests.

Autocontrast verification adds 14 tests for deterministic lazy sampling, ALEX frame
provenance, independent/shared limits, masks, nonfinite/empty/constant inputs,
float64 extremes and percentile precision, progress/cancellation and stale inputs.
They pass on Python 3.12 and 3.14. `examples/autocontrast.py` demonstrates display-only
limits while asserting original pixels are unchanged.

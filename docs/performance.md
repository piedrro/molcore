# API overhaul performance

Measured on Windows with Python 3.14.9 and NumPy 2.5.4.
Baseline source: `db67aa9099a89e6b05dfdece736108a452ae3268`; new source: the 0.2 API overhaul.
Both used the same interpreter, libraries, workload and machine, sequentially
after tests completed. These are local observations, not portable speed guarantees.

The benchmark in [examples/benchmark_api.py](../examples/benchmark_api.py) warms
imports and indexes, reports the median of five repetitions (three for photometry),
and records maximum Python-traced allocation peaks across repetitions. Peaks
exclude existing model buffers and are not process RSS or total native allocations.
The table workload has one million localisations and 200 tracks of five members.
Photometry uses a separate 20-frame movie; its expected summed intensity is 1800
under both APIs. Compressed image reads are checked against the source pixels.

| Workload | Before, ms | After, ms | Time change | Peak MiB, before / after |
| --- | ---: | ---: | --- | ---: |
| Single-ID query | 0.917 | 1.912 | 109% slower | 0.008 / 0.008 |
| Frame-range query | 116.778 | 10.963 | 10.7× faster | 1.920 / 1.920 |
| Snapshot capture | 52.112 | 4.702 | 11.1× faster | 99.334 / 0.047 |
| Single-position transaction | 69.218 | 37.286 | 1.9× faster | 103.110 / 30.578 |
| 200 short tracks | 149.726 | 207.568 | 39% slower | 0.342 / 0.420 |
| 20-frame photometry | 66.440 | 76.850 | 16% slower | 0.101 / 0.128 |
| 24 compressed HDF5 frames | 21.306 | 2.992 | 7.1× faster | 0.392 / 0.195 |
| 24 compressed TIFF frames | 262.596 | 17.766 | 14.8× faster | 0.801 / 0.358 |

Snapshot capture shares immutable scientific buffers instead of copying every
table. Transactions stage once and copy affected buffers; editing a row can still
allocate a whole column. Scoped freshness checks avoid hashing mask/image data.
Frame queries avoid constructing Python strings for every selected row in common
small-partition datasets. Bulk readers reuse handles and contiguous HDF5 reads.

The contracts add overhead to small queries and photometry: batch validation,
immutability, schemas, dependency checks and execution provenance have fixed costs.
The single-ID query, short-track query and small photometry regressions are retained
in this report. Millisecond-scale timings vary between runs; larger image workloads
may behave differently. No absolute timing threshold is enforced in tests.

Run `python examples/benchmark_api.py` to measure the current installation.
For a matched comparison, run that same script twice in the same environment with
`PYTHONPATH` pointing first to an exported baseline `src` tree and then the new
`src` tree. Do not run test suites concurrently with timing measurements.

The accompanying [raw measurements](benchmarks/api-overhaul.json) retain all
reported times and traced allocation counts.

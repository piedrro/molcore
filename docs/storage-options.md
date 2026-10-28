# Choosing snapshot image content

`SaveOptions` chooses original camera images independently from crop families.
For example, keep the original movie and photon-sequence crops:

```python
from molcore.types.storage import SaveOptions

dataset.save("snapshot.h5", options=SaveOptions(
    originals="full",
    crops=frozenset({"photon_sequence"}),
    crop_width=7,
))
```

`originals` is `none`, `full`, or `external`. `external` requires directory
layout. Crop kinds are `localization`, `track`, and `photon_sequence`; any
subset can be combined with any original-image choice. Localization and track
crops share stored localization rows. Photon crops retain independent sample
indices. Width must be a positive odd integer. Experiment snapshots accept
the same options for each dataset.

Existing `image_policy` and `crop_width` calls remain supported. Do not combine
nondefault legacy arguments with `options`. Legacy directory saves retain their
implicit external references; explicit `SaveOptions(originals="none")` omits
those references. The version-4 snapshot layout remains readable, and a new
`image_storage` attribute records the independent choices.

## Custom external sources

Register a codec once in every process that saves or opens its snapshots:

```python
from molcore.storage import register_image_source

register_image_source(
    "my_package.directory_movie.v1",
    DirectoryMovie,
    writer=lambda source: {"path": source.path, "series": source.series},
    reader=lambda descriptor: DirectoryMovie(
        descriptor["path"], series=descriptor["series"]
    ),
)
dataset.save("snapshot", layout="directory",
             options=SaveOptions(originals="external"))
```

The writer returns JSON-compatible metadata containing an absolute `path` to a
file or directory. MolCore makes it relative to the snapshot directory when
possible; the reader receives a resolved absolute path. Other descriptor fields
are passed through. Use a distinct stable kind for each descriptor schema.
The registration matches the exact source type, so subclasses must explicitly
register their own persistence semantics. `CachedSource` wrappers are unwrapped.
Built-in TIFF, FITS, and HDF5 sources use this same registry.

Only registered callbacks execute. Unknown reader kinds raise a clear error;
snapshot contents never trigger dynamic imports. Missing external paths become
`MissingSource`. A reader must return the recorded shape and dtype or opening
fails. External mode does not copy external files; retain or relocate them with
the snapshot's relative directory structure. Sources without a registered
descriptor raise an error in explicit external mode; use `originals="full"` to
embed them. Legacy directory saves retain their historical permissive behavior.

"""Picasso density rendering with explicit output geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._analysis import finish, partition, start
from ._logging import logged
from .types.analysis import RenderResult, RenderSettings

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .types import CancelEvent, CoordinateSpace, Experiment, ProgressCallback


_DEFAULT_SETTINGS = RenderSettings()


@logged
def render_localizations(
    experiment: Experiment,
    acquisition_id: str,
    channel: str,
    *,
    settings: RenderSettings = _DEFAULT_SETTINGS,
    space: CoordinateSpace | str = "raw",
    ids: ArrayLike | None = None,
    viewport: tuple[tuple[float, float], tuple[float, float]] | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: CancelEvent | None = None,
) -> RenderResult:
    """Render rows in a viewport ((y0,x0),(y1,x1)); spacing uses coordinate pixels.

    For transformed coordinates the default bounds enclose the transformed rows.
    Precision blur requires finite positive lpx/lpy in the selected coordinate
    space; affine registration propagates independent raw x/y uncertainties.

    Args:
        experiment: Scientific model supplying acquisitions and result tables.
        acquisition_id: Source acquisition identifier.
        channel: Logical source channel name.
        settings: Validated settings controlling this operation.
        space: Coordinate space: raw channel pixels, registered reference or drift-corrected reference.
        ids: Stable entity IDs; None selects the available partition.
        viewport: Viewport used by this operation.
        progress: Optional callback receiving structured progress updates.
        cancel_event: Optional event whose is_set() requests cooperative cancellation.

    Returns:
        Computed RenderResult value.
    """
    import pandas as pd
    from picasso.render import render

    expected, space = start(experiment, space, progress, cancel_event)
    view, rows = partition(experiment, acquisition_id, channel, space, ids)
    valid = np.isfinite(rows["x"]) & np.isfinite(rows["y"])
    rows = {k: v[valid] for k, v in rows.items()}
    if viewport is None:
        if space == "raw" or not len(rows["id"]):
            viewport = ((0.0, 0.0), (float(view.shape[1]), float(view.shape[2])))
        else:
            viewport = (
                (float(np.floor(rows["y"].min()) - 1), float(np.floor(rows["x"].min()) - 1)),
                (float(np.ceil(rows["y"].max()) + 1), float(np.ceil(rows["x"].max()) + 1)),
            )
    bounds = np.asarray(viewport, dtype=float)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError("Viewport must contain finite increasing y/x bounds")
    shape = np.ceil((bounds[1] - bounds[0]) * settings.oversampling).astype(int)
    if np.prod(shape.astype(float)) > 100_000_000:
        raise ValueError("Rendered image exceeds 100 million pixels; reduce oversampling or viewport")
    if settings.blur_method in ("gaussian", "gaussian_iso", "convolve") and len(rows["id"]):
        if any(k not in rows for k in ("lpx", "lpy")):
            raise ValueError("Precision blur requires lpx and lpy measurements")
        if not all(np.all(np.isfinite(rows[k]) & (rows[k] > 0)) for k in ("lpx", "lpy")):
            raise ValueError("Localization precisions must be finite and positive")
        if space != "raw":
            raw_x, raw_y = rows["lpx"].copy(), rows["lpy"].copy()
            for frame in np.unique(rows["frame"]):
                matrix = experiment.transforms.matrix(acquisition_id, channel, int(frame), space=space)[
                    :2, :2
                ]
                mask = rows["frame"] == frame
                variances = np.column_stack((raw_x[mask] ** 2, raw_y[mask] ** 2)) @ (matrix**2).T
                rows["lpx"][mask], rows["lpy"][mask] = np.sqrt(variances).T
    if len(rows["id"]):
        count, image = render(
            pd.DataFrame(rows),
            {"Pixelsize": 1.0, "Height": view.shape[1], "Width": view.shape[2]},
            disp_px_size=1 / settings.oversampling,
            viewport=bounds,
            blur_method=None if settings.blur_method == "none" else settings.blur_method,
            min_blur_width=settings.min_blur_width,
        )
    else:
        count, image = 0, np.zeros(tuple(shape), dtype=np.float32)
    finish(experiment, expected, progress, cancel_event)
    return RenderResult(
        acquisition_id,
        channel,
        space,
        settings,
        rows,
        expected,
        image,
        (float(bounds[0, 1]), float(bounds[0, 0])),
        1 / settings.oversampling,
        int(count),
    )

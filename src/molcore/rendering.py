"""Picasso density rendering with explicit output geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._analysis import finish, partition, start
from ._logging import logged
from .execution import ExecutionContext, _record_execution
from .snapshots import DatasetSnapshot, processor_model
from .transforms import project_jacobian, project_points
from .types.analysis import RenderResult, RenderSettings
from .types.selections import PartitionSelection

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .types import CoordinateSpace
_DEFAULT_SETTINGS = RenderSettings()


@logged
@_record_execution
def render_localizations(
    dataset: DatasetSnapshot,
    selection: PartitionSelection,
    *,
    settings: RenderSettings = _DEFAULT_SETTINGS,
    space: CoordinateSpace | str = "corrected",
    ids: ArrayLike | None = None,
    viewport: tuple[tuple[float, float], tuple[float, float]] | None = None,
    context: ExecutionContext | None = None,
) -> RenderResult:
    """Render rows in a viewport ((y0,x0),(y1,x1)); spacing uses coordinate pixels.

    For transformed coordinates the default bounds enclose the transformed rows.
    Precision blur requires finite positive lpx/lpy in the selected coordinate
    space; affine registration propagates independent raw x/y uncertainties.

    Args:
        selection: Explicit acquisition/channel or sequence/channel selection.
        context: Execution policy, frame chunk size, progress and cancellation.
        dataset: Scientific model supplying acquisitions and result tables.
        settings: Validated settings controlling this operation.
        space: Coordinate space; defaults to the registered, drift-corrected reference.
        ids: Stable entity IDs; None selects the available partition.
        viewport: Viewport used by this operation.
    Returns:
        Computed RenderResult value.
    """
    if not isinstance(selection, PartitionSelection):
        raise TypeError("Use PartitionSelection")
    acquisition_id, channel = (selection.acquisition_id, selection.channel)
    dataset = processor_model(dataset)
    context = context or ExecutionContext()
    context.check_cancelled()
    progress, cancel_event = (context.progress, context.cancel_event)
    import pandas as pd
    from picasso.render import render

    expected, space = start(dataset, space, progress, cancel_event, acquisition_id, (channel,))
    view, rows = partition(dataset, acquisition_id, channel, space, ids)
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
    if np.prod(shape.astype(float)) > 100000000:
        raise ValueError("Rendered image exceeds 100 million pixels; reduce oversampling or viewport")
    if settings.blur_method in ("gaussian", "gaussian_iso", "convolve") and len(rows["id"]):
        if any(k not in rows for k in ("lpx", "lpy")):
            raise ValueError("Precision blur requires lpx and lpy measurements")
        if not all(np.all(np.isfinite(rows[k]) & (rows[k] > 0)) for k in ("lpx", "lpy")):
            raise ValueError("Localization precisions must be finite and positive")
        if space != "raw":
            raw_x, raw_y = (rows["lpx"].copy(), rows["lpy"].copy())
            for frame in np.unique(rows["frame"]):
                matrix = dataset.transforms.matrix(acquisition_id, channel, int(frame), space=space)
                mask = rows["frame"] == frame
                points = project_points(
                    np.linalg.inv(matrix), np.column_stack([rows["x"][mask], rows["y"][mask]])
                )
                jacobian = project_jacobian(matrix, points)
                variances = np.einsum(
                    "nij,nj->ni", jacobian**2, np.column_stack((raw_x[mask] ** 2, raw_y[mask] ** 2))
                )
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
        count, image = (0, np.zeros(tuple(shape), dtype=np.float32))
    finish(dataset, expected, progress, cancel_event)
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

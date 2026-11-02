from pathlib import Path

root = Path("src/molcore")
p = root / "photometry.py"
s = p.read_text().replace(
    "_METRICS =", "_DEFAULT_SETTINGS = PhotometrySettings()\n\n_METRICS =", 1
)
s = s.replace(
    "settings: PhotometrySettings = PhotometrySettings()",
    "settings: PhotometrySettings = _DEFAULT_SETTINGS",
)
s = s.replace(
    "    expected = state_token(experiment)\n",
    "    if not isinstance(settings, PhotometrySettings):\n        raise TypeError('settings must be PhotometrySettings')\n    expected = state_token(experiment)\n",
    1,
)
p.write_text(s)
p = root / "tracking.py"
s = p.read_text().replace(
    "\n\ndef _job", "\n\n_DEFAULT_SETTINGS = TrackingSettings()\n\ndef _job", 1
)
s = s.replace(
    "settings: TrackingSettings = TrackingSettings()",
    "settings: TrackingSettings = _DEFAULT_SETTINGS",
)
s = s.replace(
    "        def coordinates():",
    "        def coordinates(first=first, last=last, frames=frames, subset=subset, frame_ids=frame_ids):",
)
s = s.replace(
    "    if channel not in experiment.acquisitions[acquisition_id].channels:",
    "    if not isinstance(settings, TrackingSettings):\n        raise TypeError('settings must be TrackingSettings')\n    if channel not in experiment.acquisitions[acquisition_id].channels:",
    1,
)
p.write_text(s)
p = root / "types/tables.py"
s = p.read_text().replace(
    "from numpy.typing import ArrayLike\n",
    "from numpy.typing import ArrayLike\n\nfrom ._shared import CoordinateSpace\n",
    1,
)
s = s.replace(
    '        self, *, track_id=None, frame_range=None, space="raw", sequence_id=None,',
    "        self, *, track_id: int | ArrayLike | None = None, frame_range: tuple[int, int] | None = None,\n        space: CoordinateSpace = CoordinateSpace.RAW, sequence_id: str | None = None,",
)
s = s.replace(
    "            sequence_id: Optional sequence defining frame offsets.\n",
    "            sequence_id: Optional sequence defining frame offsets.\n            acquisition_id: Optional acquisition selector; filters returned members.\n            channel: Optional logical channel selector.\n",
)
p.write_text(s)
p = root / "transforms.py"
s = p.read_text().replace(
    '        points = np.column_stack([arrays["x"], arrays["y"]]).astype(float)',
    '        if arrays["frame"].dtype.kind not in "iu" or np.any(arrays["frame"] < 0):\n            raise ValueError("Camera frames must be nonnegative integers")\n        points = np.column_stack([arrays["x"], arrays["y"]]).astype(float)',
)
p.write_text(s)
p = root / "staging.py"
s = p.read_text().replace(
    "        tuple((k, _json(a.metadata), a.pixel_size) for k, a in experiment.acquisitions.items()),",
    """        tuple((k, _json(a.metadata), a.pixel_size, _json(a.channel_specs), a.mode,
               a.excitation_sequence, a.phase, None if a.timestamps is None else a.timestamps.tobytes())
              for k, a in experiment.acquisitions.items()),""",
)
# ChannelSpec dataclasses need explicit conversion for JSON token.
s = s.replace(
    "from copy import deepcopy",
    "from copy import deepcopy\nfrom dataclasses import asdict",
)
s = s.replace(
    "_json(a.channel_specs)",
    "_json({c: asdict(spec) for c, spec in a.channel_specs.items()})",
)
s = s.replace(
    "        tuple((k, tuple(a.id for a in s.acquisitions)) for k, s in experiment.sequences.items()),",
    """        tuple((k, tuple(a.id for a in s.acquisitions), s.allow_cross_boundary_tracks, s.reference)
              for k, s in experiment.sequences.items()),""",
)
p.write_text(s)
p = root / "types/experiment.py"
s = p.read_text().replace(
    "            for values in model.processing.settings.values():\n                if isinstance(values, dict):\n                    values.pop(id, None)",
    """            for stage, values in model.processing.settings.items():
                if not isinstance(values, dict):
                    continue
                if stage == "photometry":
                    for run_id, run in list(values.items()):
                        if id in run.get("acquisitions", []):
                            run["acquisitions"].remove(id)
                            if not run["acquisitions"]:
                                del values[run_id]
                else:
                    values.pop(id, None)""",
)
p.write_text(s)
p = Path("pyproject.toml")
s = p.read_text().replace(
    '    "threadpoolctl",', '    "threadpoolctl",\n    "trackpy>=0.7,<0.8",'
)
p.write_text(s)

from pathlib import Path
p = Path('src/molcore/types/acquisition.py')
s = p.read_text().replace('        values = np.asarray(indices)\n', '        values = np.asarray(indices)\n        if values.ndim == 1 and not values.size:\n            return np.array([], dtype=np.int64)\n', 1)
s = s.replace('        values = np.asarray(frames)\n', '        values = np.asarray(frames)\n        if values.ndim == 1 and not values.size:\n            return np.array([], dtype=np.int64)\n', 1)
p.write_text(s)
p = Path('src/molcore/types/experiment.py')
s = p.read_text().replace('                         any(id in values for values in vars(model.processing).values() if isinstance(values, dict)))', '''                         any(id in values for values in model.processing.settings.values() if isinstance(values, dict)) or
                         any(id in run.get("acquisitions", []) for run in
                             model.processing.settings.get("photometry", {}).values()))''')
s = s.replace('Borrowed\n        sources are not closed; experiment-owned resources remain owned until close.', 'Borrowed\n        sources are not closed. Built-in sources reopen files per read; dropping\n        the acquisition releases its source reference without disturbing other owners.')
p.write_text(s)
p = Path('src/molcore/processing.py')
s = p.read_text() + '\n\n# Shared application entry points for detached scientific results.\nfrom .photometry import apply_photometry as apply_photometry\nfrom .tracking import apply_tracking as apply_tracking\n'
p.write_text(s)
p = Path('src/molcore/photometry.py')
s = p.read_text().replace('    columns = _columns([row for chunk in chunks for row in chunk])', "    if cancel_event is not None and cancel_event.is_set():\n        raise CancelledError('Photometry cancelled')\n    columns = _columns([row for chunk in chunks for row in chunk])")
p.write_text(s)

from pathlib import Path
p = Path('src/molcore/processing.py')
s = p.read_text().replace('\n\n# Shared application entry points for detached scientific results.\nfrom .photometry import apply_photometry as apply_photometry\nfrom .tracking import apply_tracking as apply_tracking\n', '')
s = s.replace('from .staging import transact', 'from .photometry import apply_photometry as apply_photometry\nfrom .tracking import apply_tracking as apply_tracking\nfrom .staging import transact')
p.write_text(s)
p = Path('src/molcore/types/tables.py')
s = p.read_text().replace('        tracks = list(self._members) if track_id is None else np.atleast_1d(track_id)\n        chunks = []', '''        space = CoordinateSpace(space)
        tracks = list(self._members) if track_id is None else identifiers(track_id)
        partition = None
        if acquisition_id is not None or channel is not None:
            partition = self.experiment.localizations.query(acquisition_id=acquisition_id, channel=channel)["id"]
        chunks = []''', 1)
s = s.replace('''            rows = self.experiment.localizations.query(
                ids=self._members[int(track)], frame_range=frame_range, space=space, sequence_id=sequence_id
            )
            keep = np.ones(len(rows["id"]), dtype=bool)
            if acquisition_id is not None:
                keep &= np.isin(rows["acquisition_id"], np.atleast_1d(acquisition_id))
            if channel is not None:
                keep &= np.isin(rows["channel"], np.atleast_1d(channel))
            rows = {k: v[keep] for k, v in rows.items()}''', '''            members = self._members[int(track)]
            if partition is not None:
                members = members[np.isin(members, partition)]
            if not len(members):
                continue
            rows = self.experiment.localizations.query(
                ids=members, frame_range=frame_range, space=space, sequence_id=sequence_id
            )''', 1)
s = s.replace('            rows = self.experiment.localizations.query(ids=[])\n            rows["track_id"]', '            rows = self.experiment.localizations.query(ids=[], space=space, sequence_id=sequence_id)\n            rows["track_id"]', 1)
p.write_text(s)
p = Path('tests/test_scientific_apis.py')
s = p.read_text().replace('from molcore import Acquisition, ArraySource, Experiment, StaleResultError', 'from molcore import Acquisition, ArraySource, CoordinateSpace, Experiment, StaleResultError')
s = s.replace('source_space="raw", target_space="corrected"', 'source_space=CoordinateSpace.RAW, target_space=CoordinateSpace.CORRECTED')
s = s.replace('source_space="corrected", target_space="raw"', 'source_space=CoordinateSpace.CORRECTED, target_space=CoordinateSpace.RAW')
s = s.replace("space='corrected'", 'space=CoordinateSpace.CORRECTED')
p.write_text(s)

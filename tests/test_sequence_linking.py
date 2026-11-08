import numpy as np
from scientific_helpers import compute_scenario

from molcore import Acquisition, ArraySource, ChannelSpec, Dataset, TrackingSettings
from molcore.tracking import track_sequence


class InlineExecutor:
    def map(self, operation, jobs, **options):
        return [operation(job, self) for job in jobs]

    def report(self, *args):
        pass

    def check_cancelled(self):
        pass


def test_links_across_a_concatenation_boundary_on_channel_clock():
    dataset = Dataset()
    members = []
    for acquisition_id, frames in (("first", [0, 2]), ("second", [0, 2])):
        dataset.add_acquisition(
            Acquisition(
                ArraySource(np.zeros((3, 4, 4))),
                id=acquisition_id,
                channel_specs={"dd": ChannelSpec(excitation="D")},
                excitation_sequence=("D", "A"),
            )
        )
        members.extend(
            dataset.localizations.add(
                acquisition_id=acquisition_id, channel="dd", frame=frames, x=[1, 1], y=[1, 1]
            )
        )
    sequence = dataset.add_sequence(
        ["first", "second"], allow_cross_boundary_tracks=True, reference="surface"
    )
    tracks = compute_scenario(
        track_sequence,
        dataset,
        sequence.id,
        "dd",
        cpu=InlineExecutor(),
        settings=TrackingSettings(min_length=4, frame_clock="channel"),
    )
    assert len(tracks.memberships) == 1
    np.testing.assert_array_equal(tracks.memberships[0], members)
    rows = dataset.tracks.query(sequence_id=sequence.id)
    np.testing.assert_array_equal(rows["id"], members)
    np.testing.assert_array_equal(rows["frame"], [0, 2, 0, 2])

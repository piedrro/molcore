from dataclasses import replace

import numpy as np
import pytest

from molcore import Acquisition, ArraySource, BoundingBox, Dataset, StaleResultError
from molcore.photometry import FixedApertureSettings, bounding_box_traces


def model():
    dataset = Dataset()
    dataset.add_acquisition(Acquisition(ArraySource(np.ones((3, 9, 9))), id="movie"))
    box = dataset.bounding_boxes.set(
        BoundingBox(
            id="site",
            x=4,
            y=4,
            width=5,
            height=5,
            acquisition_id="movie",
            channel="image",
            metadata={"origin": "manual"},
        )
    )
    return dataset, box


def test_box_edits_preserve_identity_and_invalidate_dependent_traces():
    dataset, box = model()
    result = bounding_box_traces(dataset, "movie", "image", [box.id])
    sequence = dataset.photon_sequences[result.sequence_ids[0]]
    assert sequence.bounding_box_id == box.id
    assert sequence.bounding_box_revision == box.revision
    assert box.created_at and box.updated_at
    dataset.bounding_boxes.set(replace(box, x=5))
    assert dataset.bounding_boxes[box.id].revision == box.revision + 1
    assert dataset.bounding_boxes[box.id].created_at == box.created_at
    assert dataset.photon_sequences.is_stale(sequence.id)
    assert sequence.metadata["bounding_box"]["x"] == 4


def test_stale_box_preview_cannot_be_applied():
    dataset, box = model()
    result = bounding_box_traces(dataset, "movie", "image", [box.id], apply=False)
    dataset.bounding_boxes.set(replace(box, width=7))
    with pytest.raises(StaleResultError):
        dataset.apply(result)
    assert not dataset.photon_sequences.query()


def test_box_metadata_and_extraction_provenance_roundtrip_without_pixels(tmp_path):
    dataset, box = model()
    result = bounding_box_traces(
        dataset,
        "movie",
        "image",
        [box.id],
        settings=FixedApertureSettings(spot_size=3, buffer=1, global_background=True),
    )
    path = tmp_path / "boxes.h5"
    dataset.save(path, image_policy="none")
    restored = Dataset.open(path)
    assert restored.bounding_boxes[box.id] == box
    sequence = restored.photon_sequences[result.sequence_ids[0]]
    assert sequence.bounding_box_id == box.id
    assert sequence.metadata["extraction"]["settings"]["spot_size"] == 3
    assert sequence.metadata["extraction"]["source_space"] == "raw"
    assert sequence.metadata["extraction"]["software"]["molcore"]
    restored.bounding_boxes.remove(box.id)
    assert restored.photon_sequences.is_stale(sequence.id)
    assert restored.photon_sequences[sequence.id].metadata["bounding_box"]["metadata"] == {"origin": "manual"}


def test_working_boxes_require_an_explicit_working_source():
    dataset, _ = model()
    box = dataset.bounding_boxes.set(BoundingBox(x=4, y=4, width=3, height=3, space="working"))
    with pytest.raises(ValueError, match="working source"):
        bounding_box_traces(dataset, "movie", "image", [box.id])
    result = bounding_box_traces(
        dataset,
        "movie",
        "image",
        [box.id],
        source=ArraySource(np.ones((3, 9, 9))),
        apply=False,
    )
    assert result.sequences[0].positions is None


def test_snapshot_box_edits_do_not_change_the_live_store():
    dataset, box = model()
    snapshot = dataset.snapshot()
    snapshot.bounding_boxes.set(replace(box, y=6))
    assert dataset.bounding_boxes[box.id].y == 4
    assert snapshot.bounding_boxes[box.id].y == 6


def test_noop_box_updates_and_detached_metadata_preserve_active_sequences():
    dataset, box = model()
    result = bounding_box_traces(dataset, "movie", "image", [box.id])
    revision = dataset.bounding_boxes.revision
    dataset.bounding_boxes.set(box)
    detached = dataset.bounding_boxes[box.id]
    detached.metadata["origin"] = "outside edit"
    assert dataset.bounding_boxes[box.id].metadata["origin"] == "manual"
    assert dataset.bounding_boxes.revision == revision
    assert not dataset.photon_sequences.is_stale(result.sequence_ids[0])


def test_editing_a_neighbour_invalidates_the_shared_background_run():
    dataset, box = model()
    neighbour = dataset.bounding_boxes.set(replace(box, id="neighbour", x=6))
    result = bounding_box_traces(
        dataset,
        "movie",
        "image",
        [box.id, neighbour.id],
        settings=FixedApertureSettings(),
    )
    dataset.bounding_boxes.remove(neighbour.id)
    assert all(dataset.photon_sequences.is_stale(identifier) for identifier in result.sequence_ids)


def test_deleted_box_history_roundtrips_and_recreation_advances_revision(tmp_path):
    dataset, box = model()
    result = bounding_box_traces(dataset, "movie", "image", [box.id])
    dataset.bounding_boxes.remove(box.id)
    path = tmp_path / "deleted.h5"
    dataset.save(path, image_policy="none")
    restored = Dataset.open(path)
    assert not len(restored.bounding_boxes)
    assert restored.photon_sequences.is_stale(result.sequence_ids[0])
    recreated = restored.bounding_boxes.set(box)
    assert recreated.revision > box.revision
    assert restored.photon_sequences.is_stale(result.sequence_ids[0])


def test_acquisition_removal_requires_cascade_for_bound_boxes(tmp_path):
    dataset, box = model()
    with pytest.raises(ValueError, match="cascade"):
        dataset.remove_acquisition("movie")
    dataset.remove_acquisition("movie", cascade=True)
    assert not len(dataset.bounding_boxes)
    path = tmp_path / "removed.h5"
    dataset.save(path, image_policy="none")
    restored = Dataset.open(path)
    assert not len(restored.bounding_boxes)
    assert restored.bounding_boxes.query(include_deleted=True)[0].id == box.id

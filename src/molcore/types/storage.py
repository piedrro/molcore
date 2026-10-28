"""Independent choices for original images and derived image crops."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SaveOptions:
    """Select original image retention independently of crop families.

    External references require directory layout and a registered source writer.
    Crop width is an odd, positive number of camera pixels.
    """

    originals: Literal["none", "full", "external"] = "none"
    crops: frozenset[Literal["localization", "track", "photon_sequence"]] = frozenset()
    crop_width: int = 7

    def __post_init__(self) -> None:
        from ..crops import validate_width

        if self.originals not in ("none", "full", "external"):
            raise ValueError("originals must be none, full or external")
        kinds = frozenset(self.crops)
        if kinds - {"localization", "track", "photon_sequence"}:
            raise ValueError("Unsupported crop kind")
        object.__setattr__(self, "crops", kinds)
        validate_width(self.crop_width)

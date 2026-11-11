"""Supported fiducial transform models and their minimum sample sizes."""

MINIMUM_PAIRS = {"translation": 1, "rigid": 2, "similarity": 2, "affine": 3, "homography": 4}
MODEL_CHOICES = ", ".join(MINIMUM_PAIRS)

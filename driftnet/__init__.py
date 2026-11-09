"""DriftNet: Dual-Head Trajectory Transformer & Real-Time Streaming Monitor."""

import sys

try:
    import torchvision  # noqa: F401
    from torchvision.transforms import InterpolationMode  # noqa: F401
    _ = InterpolationMode.NEAREST_EXACT
except Exception:
    # Avoid legacy torchvision binary/attribute conflicts when importing transformers
    sys.modules["torchvision"] = None

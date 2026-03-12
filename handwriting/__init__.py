"""Handwriting pipeline package."""

__all__ = ["preprocess", "segmentation", "export_text"]


def normalize_path(path: str) -> str:
    """Normalize path separators to forward slashes for cross-platform CSV storage."""
    return path.replace("\\", "/")

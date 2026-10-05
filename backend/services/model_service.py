import os
import sys
from functools import lru_cache


def _ensure_repo_root_on_path() -> None:
    """Allow the backend to import the sibling model package."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


@lru_cache(maxsize=1)
def _get_predictor():
    """Load the real model predictor once per backend process."""
    _ensure_repo_root_on_path()
    from model.predict import predict
    return predict


def predict(video_path: str) -> dict:
    """Run the trained MSLI model on a saved video."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    result = _get_predictor()(video_path)

    if not result.get("success"):
        raise RuntimeError(result.get("error") or "Model prediction failed")

    return {
        "sign": result["sign"],
        "confidence": float(result["confidence"]),
        "top_predictions": result.get("top_predictions", []),
    }

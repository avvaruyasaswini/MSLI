import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_predictor():
    """Load the real model predictor once per backend process."""
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

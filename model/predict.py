import os
import json
import numpy as np
import keras
from keras import initializers
from keras.layers import Layer

# --- Universal Deserialization Patch ---
# Intercept configs at the base level before __init__ is called
_orig_layer_from_config = Layer.from_config
_unsupported_layer_keys = {
    "quantization_config",
    "renorm",
    "renorm_clipping",
    "renorm_momentum",
    "synchronized",
}

@classmethod
def _safe_layer_from_config(cls, config):
    config = config.copy()
    for key in _unsupported_layer_keys:
        config.pop(key, None)
    return _orig_layer_from_config.__func__(cls, config)

Layer.from_config = _safe_layer_from_config

_orig_init_from_config = initializers.Initializer.from_config
_unsupported_init_keys = {"input_axes", "output_axes"}

@classmethod
def _safe_init_from_config(cls, config):
    config = config.copy()
    for key in _unsupported_init_keys:
        config.pop(key, None)
    return _orig_init_from_config.__func__(cls, config)

initializers.Initializer.from_config = _safe_init_from_config
# ---------------------------------------

MODEL_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_ROOT, "checkpoints", "isl_model_best.keras")
CLASS_MAP_PATH = os.path.join(MODEL_ROOT, "checkpoints", "class_map.json")

_MODEL = None
_CLASS_MAP = None


def _load_artifacts():
    """Lazily loads the trained model weights and class mapping once."""
    global _MODEL, _CLASS_MAP
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model checkpoint not found at: {MODEL_PATH}. "
                "Ensure trained weights are placed in model/checkpoints/."
            )
        _MODEL = keras.models.load_model(MODEL_PATH, compile=False)

    if _CLASS_MAP is None:
        if not os.path.exists(CLASS_MAP_PATH):
            raise FileNotFoundError(
                f"Class mapping file not found at: {CLASS_MAP_PATH}. "
                "Ensure class_map.json exists in model/checkpoints/."
            )
        with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
            _CLASS_MAP = {int(k): v for k, v in raw_map.items()}


def predict(input_data, flip_horizontal: bool = False, top_k: int = 5) -> dict:
    """
    Runs model inference.
    Accepts either:
      - np.ndarray of shape (32, 225) or (1, 32, 225)
      - str: path to an MP4 video clip (extracts landmarks automatically)
    """
    _load_artifacts()

    # If input is a file path string, extract landmarks first
    if isinstance(input_data, str):
        if not os.path.exists(input_data):
            return {
                "success": False,
                "sign": "",
                "confidence": 0.0,
                "top_predictions": [],
                "error": f"Video file not found at: {input_data}"
            }
        from model.src.extract_landmarks import extract_landmarks_from_video
        sequence = extract_landmarks_from_video(input_data, flip_horizontal=flip_horizontal)
    else:
        sequence = input_data

    # Validate sequence type
    if not isinstance(sequence, np.ndarray):
        return {
            "success": False,
            "sign": "",
            "confidence": 0.0,
            "top_predictions": [],
            "error": f"Expected numpy array for sequence, got {type(sequence).__name__}"
        }

    # Expand dims if single sequence without batch dimension
    if sequence.ndim == 2:
        sequence = np.expand_dims(sequence, axis=0)

    predictions = _MODEL.predict(sequence, verbose=0)[0]
    
    # Sort probabilities in descending order
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_predictions = [
        {"sign": _CLASS_MAP.get(int(i), "UNKNOWN"), "confidence": float(predictions[i])}
        for i in top_indices
    ]

    best_idx = int(top_indices[0])
    best_confidence = float(predictions[best_idx])
    predicted_sign = _CLASS_MAP.get(best_idx, "UNKNOWN")

    return {
        "success": True,
        "sign": predicted_sign,
        "confidence": best_confidence,
        "top_predictions": top_predictions,
        "error": None
    }
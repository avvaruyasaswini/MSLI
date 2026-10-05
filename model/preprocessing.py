import numpy as np
import pandas as pd

MAX_FRAMES = 32
NUM_FEATURES = 225

# The canonical feature order used by the project.
# Keep this explicit; do not rely on alphabetical sorting of the type column.
LANDMARK_LAYOUT = (
    ("left_hand", 21),
    ("pose", 33),
    ("right_hand", 21),
)


def frame_from_landmark_group(group: pd.DataFrame) -> np.ndarray | None:
    """Build one 225-D frame in the canonical LH | Pose | RH order."""
    parts = []

    for landmark_type, expected_count in LANDMARK_LAYOUT:
        part = group[group["type"] == landmark_type].sort_values("landmark_index")
        if len(part) != expected_count:
            return None

        coords = part[["x", "y", "z"]].to_numpy(dtype=np.float32)
        if coords.shape != (expected_count, 3):
            return None

        parts.append(coords.reshape(-1))

    frame = np.concatenate(parts).astype(np.float32)
    return np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_frame_coordinates(coords: np.ndarray) -> np.ndarray:
    """Normalize x/y around the pose shoulder midpoint.

    Feature layout is LH | Pose | RH, so:
      left shoulder  = 63 + 11*3 = 96
      right shoulder = 63 + 12*3 = 99
    """
    if coords.shape != (NUM_FEATURES,):
        raise ValueError(f"Expected ({NUM_FEATURES},), got {coords.shape}")

    ls_x, ls_y = coords[96], coords[97]
    rs_x, rs_y = coords[99], coords[100]

    if (ls_x != 0.0 or ls_y != 0.0) and (rs_x != 0.0 or rs_y != 0.0):
        anchor_x = (ls_x + rs_x) / 2.0
        anchor_y = (ls_y + rs_y) / 2.0
        torso_scale = float(np.hypot(ls_x - rs_x, ls_y - rs_y))
        if torso_scale < 1e-4:
            torso_scale = 1.0
    else:
        anchor_x, anchor_y, torso_scale = 0.5, 0.5, 1.0

    normalized = coords.copy()
    for i in range(0, NUM_FEATURES, 3):
        if coords[i] != 0.0 or coords[i + 1] != 0.0:
            normalized[i] = (coords[i] - anchor_x) / torso_scale
            normalized[i + 1] = (coords[i + 1] - anchor_y) / torso_scale

    return normalized


def resample_sequence(
    frames: np.ndarray,
    target_frames: int = MAX_FRAMES,
) -> np.ndarray:
    """Resample or zero-pad a sequence to a fixed number of frames."""
    if frames.size == 0:
        return np.zeros((target_frames, NUM_FEATURES), dtype=np.float32)

    n = len(frames)
    if n == target_frames:
        return frames.astype(np.float32)
    if n > target_frames:
        indices = np.linspace(0, n - 1, target_frames, dtype=int)
        return frames[indices].astype(np.float32)

    return np.pad(
        frames.astype(np.float32),
        ((0, target_frames - n), (0, 0)),
        mode="constant",
    )


def preprocess_sequence(
    frames: list[np.ndarray] | np.ndarray,
    target_frames: int = MAX_FRAMES,
    normalize: bool = False,
) -> np.ndarray:
    """Canonical preprocessing shared by training and inference.

    IMPORTANT: normalize=False is intentional for the currently deployed
    isl_model_best.keras checkpoint, which was trained on raw coordinates.
    A newly trained normalized model must set normalize=True in both paths.
    """
    if not frames:
        return np.zeros((target_frames, NUM_FEATURES), dtype=np.float32)

    seq = np.asarray(frames, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != NUM_FEATURES:
        raise ValueError(
            f"Expected sequence with shape (N, {NUM_FEATURES}), got {seq.shape}"
        )

    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize:
        seq = np.asarray(
            [normalize_frame_coordinates(frame) for frame in seq],
            dtype=np.float32,
        )

    return resample_sequence(seq, target_frames)


def parse_parquet(file_path: str, normalize: bool = False) -> np.ndarray:
    """Read one labeled parquet sample using the canonical feature order."""
    df = pd.read_parquet(file_path)
    df = df[df["type"].isin([name for name, _ in LANDMARK_LAYOUT])]

    frames = []
    for _, group in df.groupby("frame", sort=True):
        frame = frame_from_landmark_group(group)
        if frame is not None:
            frames.append(frame)

    return preprocess_sequence(frames, normalize=normalize)

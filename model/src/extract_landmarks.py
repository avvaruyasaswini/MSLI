import cv2
import mediapipe as mp
import numpy as np

from model.preprocessing import (
    MAX_FRAMES,
    NUM_FEATURES,
    normalize_frame_coordinates,
    preprocess_sequence,
)

mp_holistic = mp.solutions.holistic

# Keep this False while using the existing checkpoint. The checkpoint was
# trained on raw MediaPipe coordinates. A newly trained normalized model must
# use the same setting here and in model/train.py.
NORMALIZE_COORDINATES = False


def _landmarks_or_zeros(landmarks, count: int) -> np.ndarray:
    if not landmarks:
        return np.zeros(count * 3, dtype=np.float32)

    return np.asarray(
        [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
        dtype=np.float32,
    ).reshape(-1)


def _build_raw_frame(results) -> np.ndarray:
    lh = _landmarks_or_zeros(results.left_hand_landmarks, 21)
    pose = _landmarks_or_zeros(results.pose_landmarks, 33)
    rh = _landmarks_or_zeros(results.right_hand_landmarks, 21)
    return np.concatenate([lh, pose, rh]).astype(np.float32)


def extract_landmarks_from_video(
    video_path: str,
    target_frames: int = MAX_FRAMES,
    flip_horizontal: bool = False,
) -> np.ndarray:
    """Extract exactly the same feature layout/temporal sampling used in training."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        cap.release()
        return np.zeros((target_frames, NUM_FEATURES), dtype=np.float32)

    try:
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        ) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if flip_horizontal:
                    frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_frame)

                frames.append(_build_raw_frame(results))
    finally:
        cap.release()

    # No activity trimming, persistence, or backfill here.
    # Those were inference-only transformations and did not exist in the
    # checkpoint's training preprocessing.
    return preprocess_sequence(
        frames,
        target_frames=target_frames,
        normalize=NORMALIZE_COORDINATES,
    )

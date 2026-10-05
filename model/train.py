import os
import glob
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

from model.preprocessing import (
    MAX_FRAMES,
    NUM_FEATURES,
    parse_parquet,
)

DATA_DIR = "model/data/dataset_263/keypoints"
CHECKPOINT_DIR = "model/checkpoints"
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "isl_model_best.keras")
CLASS_MAP_PATH = os.path.join(CHECKPOINT_DIR, "class_map.json")
CACHE_FILE = "model/data/preprocessed_data_raw.npz"

# This must stay False for the currently deployed checkpoint.
# If you later retrain a normalized model, switch this to True and keep
# the same value in model/src/extract_landmarks.py.
NORMALIZE_COORDINATES = False

BATCH_SIZE = 64
EPOCHS = 40

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# --- 1. Load or Generate Cached Data ---
if os.path.exists(CACHE_FILE):
    print(">> Loading canonical raw-coordinate data from cache...")
    cache = np.load(CACHE_FILE)
    X, y, classes = cache["X"], cache["y"], cache["classes"]
else:
    print(">> Processing parquet files with canonical LH | Pose | RH ordering...")
    parquet_files = glob.glob(f"{DATA_DIR}/*/*/*.parquet")
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {DATA_DIR}")

    classes = sorted(
        list(set(os.path.basename(os.path.dirname(f)) for f in parquet_files))
    )
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    X_list, y_list = [], []
    total = len(parquet_files)

    for idx, file_path in enumerate(parquet_files):
        if (idx + 1) % 500 == 0 or (idx + 1) == total:
            print(
                f"   [{idx + 1}/{total}] Processing: "
                f"{os.path.basename(file_path)}"
            )

        label = os.path.basename(os.path.dirname(file_path))
        feats = parse_parquet(
            file_path,
            normalize=NORMALIZE_COORDINATES,
        )
        X_list.append(feats)
        y_list.append(class_to_idx[label])

    X = np.nan_to_num(
        np.asarray(X_list, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    y = np.asarray(y_list, dtype=np.int32)

    print(f">> Saving cache to {CACHE_FILE}...")
    np.savez_compressed(CACHE_FILE, X=X, y=y, classes=classes)
    print(">> Cache saved successfully.")

# Save class mapping
idx_to_class = {int(i): str(cls_name) for i, cls_name in enumerate(classes)}
with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(idx_to_class, f, indent=2)

num_classes = len(classes)
print(
    f">> Dataset ready: {len(X)} samples, {num_classes} classes. "
    f"NaNs remaining: {np.isnan(X).any()}"
)

# --- 2. Train/Validation Split ---
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# --- 3. Sequence Model Architecture ---
model = models.Sequential([
    layers.Input(shape=(MAX_FRAMES, NUM_FEATURES)),
    layers.BatchNormalization(),
    layers.Bidirectional(layers.GRU(128, return_sequences=True)),
    layers.Dropout(0.3),
    layers.Bidirectional(layers.GRU(64)),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# --- 4. Callbacks & Training ---
cb = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    ),
    callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
    ),
]

print(
    ">> Starting model training with canonical "
    f"LH | Pose | RH preprocessing (normalize={NORMALIZE_COORDINATES})..."
)
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=cb,
)

print(f"\n>> Training complete! Model saved to: {MODEL_SAVE_PATH}")

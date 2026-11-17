# Kaggle cell-wise codes

## Result of the codebase scan

The project is **not complete as supplied**. `train.py`, `Angel-I-Baby-Monitor.py`, and
`Train.ipynb` contain the same Colab-exported training/visualization code, but they
cannot be run end-to-end without changes:

- The scripts use Google Colab paths such as `/content/drive/...`, not Kaggle paths.
- The validation path is inconsistent (`dataset1/train` versus `data/validation`).
- The first emotion archive contains seven training classes, but its `validation/`
  directory is empty.
- The second emotion archive contains only `angry`, `fear`, `happy`, and `neutral`;
  it does not contain all seven output classes.
- `keras.preprocessing.image.ImageDataGenerator`, `Adam(lr=...)`, and
  `fit_generator(...)` are obsolete or incompatible in current Kaggle TensorFlow
  environments.
- The later cells reference `emotion_model_new.json` and
  `emotion_model_new.h5`, but no model artifacts are included in this project.
- The later cells are optional model visualization/ONNX/Colab-server experiments,
  not required to train the model. The `caffe` and `ngrok` cells are not part of a
  portable Kaggle training workflow.
- `Angel-I-Baby-Monitor.py` is byte-for-byte identical to `train.py`; it does not
  contain a separate baby-monitor inference application.
- Three ZIP files are present locally, but only two Kaggle dataset paths were
  supplied. This guide uses exactly the two paths supplied below and does not
  invent a path for the third archive.

The cells below reconstruct the trainable seven-class emotion model. They accept
either extracted dataset folders or ZIP files stored below the exact Kaggle dataset
roots. They use the first archive's seven-class training data and use a reproducible
validation split from that training data because the supplied validation directory
is empty. The second archive is included as additional training data where its
classes overlap.

> **Before running:** verify that these two exact roots contain the uploaded data
> (or the corresponding ZIP files). If the third ZIP is intended to be used, its
> exact Kaggle path is still required.

---

## Cell 1 — Imports and exact Kaggle dataset roots

```python
from pathlib import Path
import shutil
import zipfile

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

DATASET1_ROOT = Path("/kaggle/input/datasets/prishabhkumar/dataset1")
DATASET2_ROOT = Path("/kaggle/input/datasets/prishabhkumar/dataset2")
WORK_ROOT = Path("/kaggle/working/emotion_data")
MODEL_ROOT = Path("/kaggle/working/model")

for root in (DATASET1_ROOT, DATASET2_ROOT):
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

print("TensorFlow:", tf.__version__)
print("Dataset 1:", DATASET1_ROOT)
print("Dataset 2:", DATASET2_ROOT)
```

## Cell 2 — Extract uploaded ZIP files when necessary

```python
def prepare_dataset_root(source_root: Path, destination: Path) -> Path:
    """Return a directory containing the uploaded dataset contents."""
    destination.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(source_root.rglob("*.zip"))
    if zip_files:
        for archive in zip_files:
            archive_destination = destination / archive.stem
            archive_destination.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zipped:
                zipped.extractall(archive_destination)
        return destination

    for item in source_root.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    return destination


SOURCE1 = prepare_dataset_root(DATASET1_ROOT, WORK_ROOT / "source1")
SOURCE2 = prepare_dataset_root(DATASET2_ROOT, WORK_ROOT / "source2")
print("Prepared:", SOURCE1)
print("Prepared:", SOURCE2)
```

## Cell 3 — Locate the extracted train directories

```python
def find_train_directory(root: Path) -> Path:
    candidates = sorted(
        path for path in root.rglob("train")
        if path.is_dir() and any(child.is_dir() for child in path.iterdir())
    )
    if not candidates:
        raise FileNotFoundError(f"No class-based train directory found below {root}")
    return candidates[0]


TRAIN1 = find_train_directory(SOURCE1)
TRAIN2 = find_train_directory(SOURCE2)
print("Training directory 1:", TRAIN1)
print("Training directory 2:", TRAIN2)
print("Classes 1:", sorted(path.name for path in TRAIN1.iterdir() if path.is_dir()))
print("Classes 2:", sorted(path.name for path in TRAIN2.iterdir() if path.is_dir()))
```

## Cell 4 — Combine the two training sources

```python
COMBINED_TRAIN = WORK_ROOT / "combined_train"
if COMBINED_TRAIN.exists():
    shutil.rmtree(COMBINED_TRAIN)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

for source_index, train_directory in enumerate((TRAIN1, TRAIN2), start=1):
    for class_directory in sorted(path for path in train_directory.iterdir() if path.is_dir()):
        destination_class = COMBINED_TRAIN / class_directory.name
        destination_class.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(class_directory.iterdir()):
            if image_path.suffix.lower() not in image_extensions:
                continue
            destination = destination_class / f"source{source_index}_{image_path.name}"
            shutil.copy2(image_path, destination)

classes = sorted(path.name for path in COMBINED_TRAIN.iterdir() if path.is_dir())
if len(classes) != 7:
    raise ValueError(f"Expected seven emotion classes, found {len(classes)}: {classes}")

print("Combined classes:", classes)
for class_name in classes:
    count = sum(1 for path in (COMBINED_TRAIN / class_name).iterdir() if path.is_file())
    print(f"{class_name}: {count}")
```

## Cell 5 — Create training and validation generators

```python
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 64
SEED = 42

data_generator = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
)

train_generator = data_generator.flow_from_directory(
    COMBINED_TRAIN,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training",
    seed=SEED,
    shuffle=True,
)

validation_generator = data_generator.flow_from_directory(
    COMBINED_TRAIN,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation",
    seed=SEED,
    shuffle=False,
)

print("Class mapping:", train_generator.class_indices)
```

## Cell 6 — Build and compile the CNN

```python
cv2.ocl.setUseOpenCL(False)

emotion_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)),
    Conv2D(64, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(128, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(512, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(7, activation="softmax"),
])

emotion_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.0001),
    metrics=["accuracy"],
)
emotion_model.summary()
```

## Cell 7 — Train

```python
history = emotion_model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
)
```

## Cell 8 — Plot training history

```python
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.title("Model accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Model loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
```

## Cell 9 — Evaluate and save model artifacts

```python
validation_loss, validation_accuracy = emotion_model.evaluate(validation_generator)
print(f"Validation loss: {validation_loss:.4f}")
print(f"Validation accuracy: {validation_accuracy:.4f}")

MODEL_ROOT.mkdir(parents=True, exist_ok=True)
emotion_model.save(MODEL_ROOT / "emotion_model.keras")
(MODEL_ROOT / "class_indices.txt").write_text(
    repr(train_generator.class_indices),
    encoding="utf-8",
)
print("Saved model artifacts to:", MODEL_ROOT)
```

## What is still missing for a complete application

These cells train and save the classifier, but the supplied codebase does not
contain a real inference loop, webcam/audio integration, alarm logic, deployment
configuration, requirements file, or the pre-trained `emotion_model_new` files
referenced by the old visualization cells. Those requirements must be provided
before the project can be called a complete baby-monitor application.

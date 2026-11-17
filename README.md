# Infant Expression Recognition

> An image-classification research project for understanding infant expressions with computer vision.

Angel-I explores how facial-expression datasets can be cleaned, audited, combined, and used to train convolutional neural networks for two related tasks:

1. **General emotion recognition** across seven classes: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`.
2. **Infant-expression recognition** across four classes: `angry`, `cry`, `laugh`, and `normal`.

The project is currently notebook-led. Its main notebook investigates dataset quality and trains both a shared multi-task CNN and a dedicated infant-only CNN. The repository also preserves an earlier Google Colab/Keras experiment for reference.

## What the project does

The main workflow in [`Notebooks/new-image-project.ipynb`](./Notebooks/new-image-project.ipynb) covers:

- discovering and inspecting the available datasets;
- counting images by source and class;
- visually auditing representative samples;
- checking image metadata, dimensions, and documentation;
- detecting exact duplicates and possible near-duplicates;
- checking for cross-dataset and cross-class leakage;
- building a cleaned manifest and reproducible train/validation splits;
- preparing PyTorch datasets and data loaders;
- training a shared CNN backbone with task-specific classification heads;
- evaluating both tasks with accuracy, classification reports, and confusion matrices;
- comparing learned representations between the two tasks; and
- training an infant-only CNN as a focused alternative.

The project is intended as a research and experimentation foundation. It does **not** yet contain a production webcam application, audio pipeline, alerting system, or packaged inference service.

## Repository layout

```text
.
├── Assets/
│   └── infant_expression_final_assets.zip
├── Data/
│   └── Dataset archives and extracted dataset material
├── Notebooks/
│   └── new-image-project.ipynb       # Main audit and PyTorch experiment
├── code/code/
│   ├── Angel-I-Baby-Monitor.py       # Legacy Colab/Keras export
│   ├── train.py                      # Legacy Keras training/export script
│   ├── Train.ipynb                   # Legacy training notebook
│   └── docs/
│       └── cell_wise_codes.md        # Reconstructed portable training cells
├── .gitignore
└── README.md
```

Large dataset archives are intentionally kept out of normal Git tracking. See [`.gitignore`](./.gitignore) for the repository rules.

## Dataset sources

The notebooks were written for a Kaggle environment with these dataset roots:

```text
/kaggle/input/datasets/prishabhkumar/dataset1
/kaggle/input/datasets/prishabhkumar/dataset2
/kaggle/input/datasets/prishabhkumar/mendely-dataset
```

The three sources are used differently:

| Source | Role in the project |
| --- | --- |
| Dataset 1 | Seven-class facial-expression data with `train` and `test` structure |
| Dataset 2 | Additional emotion data with overlapping classes |
| Mendeley | Infant-expression images in primary and cropped forms |

The exact archive contents and directory names should be verified before training. The repository contains several large ZIP files, but the notebook expects the Kaggle dataset layout above rather than a fixed local extraction path.

## Getting started

### Recommended environment

The main experiment is designed for a notebook runtime such as Kaggle or Google Colab with access to a GPU. A practical environment includes:

- Python 3.10 or newer;
- PyTorch and Torchvision;
- NumPy and pandas;
- Pillow and OpenCV;
- scikit-learn;
- Matplotlib and Seaborn; and
- tqdm.

Because the repository does not currently include a pinned `requirements.txt` or `environment.yml`, install versions compatible with the notebook runtime you choose.

### Run the main experiment

1. Upload or attach the three datasets to the notebook environment.
2. Confirm that the Kaggle paths in [`new-image-project.ipynb`](./Notebooks/new-image-project.ipynb) exist.
3. Open the notebook and run the cells from top to bottom.
4. Review the duplicate/leakage audit before trusting model metrics.
5. The best multi-task checkpoint is written to the notebook runtime at:

   ```text
   /kaggle/working/checkpoints/best_multitask_infant_cnn.pth
   ```

   The infant-only experiment uses:

   ```text
   /kaggle/working/checkpoints/best_infant_only_cnn.pth
   ```

The notebook uses a fixed random seed in its data-preparation workflow where applicable, but results can still vary with hardware, library versions, and dataset revisions.

## Model design

### Multi-task CNN

The main experiment uses a shared convolutional feature extractor with two output heads:

```text
image
  └── shared CNN backbone
        ├── seven-class emotion head
        └── four-class infant-expression head
```

This allows the model to learn general facial features while keeping the label spaces separate. During training, the two tasks are evaluated independently and the best checkpoint is selected using infant-expression validation loss.

### Infant-only CNN

The notebook also defines a separate lightweight CNN dedicated to the four infant-expression classes. It uses depthwise-separable blocks, adaptive average pooling, dropout, AdamW optimization, and a `ReduceLROnPlateau` scheduler.

## Legacy Keras experiment

The files in [`code/code`](./code/code) are an earlier Colab export. They build a grayscale Keras CNN using `48 × 48` inputs and a seven-class softmax output, then include optional model export and visualization experiments.

They are useful as historical reference, but they are not currently plug-and-play:

- paths point to `/content/drive/...` Google Drive locations;
- the validation path is inconsistent with the training path;
- the scripts contain notebook-only commands such as `!pip install`;
- several APIs are obsolete in current TensorFlow/Keras versions; and
- referenced model files such as `emotion_model_new.json` and `emotion_model_new.h5` are not included.

For a more portable version of the seven-class training flow, see [`cell_wise_codes.md`](./code/code/docs/cell_wise_codes.md).

## Data-quality considerations

The dataset audit is a central part of this project. Before interpreting validation scores, inspect:

- class imbalance across all sources;
- duplicate images and near-duplicate images;
- the relationship between primary and cropped Mendeley images;
- possible leakage between training and validation sets;
- inconsistent image dimensions and color modes; and
- differences in label definitions between general emotion and infant-expression datasets.

In particular, a high score on a random image split may not represent real-world performance if visually related images from the same subject or capture session appear in both splits.

## Current limitations

- No committed model checkpoint is included.
- No inference script or command-line prediction interface is included.
- No webcam, video, audio, or notification integration is implemented.
- Dataset archives are large and are not intended to be loaded directly by Git.
- Training paths are notebook-environment specific and require adjustment for local execution.
- The project has no pinned dependency file or automated test suite yet.

## Roadmap

- add a reproducible local data-preparation script;
- add a pinned environment and configuration file;
- export the trained model together with its class mapping;
- implement image, webcam, and video inference;
- add subject-aware splitting to reduce leakage risk;
- report per-class precision, recall, F1, and calibration; and
- package the infant-expression model behind a small application or API.

## Responsible use

Expression classifiers are probabilistic tools, not reliable measures of an infant's wellbeing, intent, pain, or safety. Predictions should not replace caregiver observation or professional medical advice. Any deployment involving children should prioritize privacy, consent, secure data handling, and careful validation across diverse conditions.

## License and dataset attribution

No license or dataset attribution file is currently included. Before publishing or redistributing the code, model, or datasets, add the appropriate project license and follow the original terms for every dataset source.

# ISIC 2019 Classification Baseline

An eight-class skin-lesion classification baseline using an ImageNet-pretrained EfficientNet-B0. This project prepares a baseline for future active-learning experiments; an active-learning loop is not implemented yet.

## Method

- Classes: MEL, NV, BCC, AK, BKL, DF, VASC, and SCC.
- Excludes ground-truth rows with no positive label among these eight classes.
- Splits lesion groups approximately 70/15/15 into training, validation, and test sets, stratified by diagnosis. Images without a lesion ID are treated as separate groups, so grouping cannot protect against unknown same-lesion relationships.
- Uses inverse-square-root class weights, classifier-head training, and optional backbone fine-tuning.
- Selects model weights using validation macro-F1 and evaluates on the held-out test split.

## Setup

Create a Python environment and install dependencies. PowerShell example from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the ISIC 2019 training images, training ground truth, and training metadata separately and arrange them as follows:

```text
data/
  train/
    ISIC_0000000.jpg
    ...
  ISIC_2019_Training_GroundTruth.csv
  ISIC_2019_Training_Metadata.csv
```

Datasets, virtual environments, model weights, and generated outputs are excluded from Git. Follow the dataset's own licensing and attribution requirements.

## Quick preliminary classification run

Train the classifier head for one epoch, skip backbone fine-tuning, and evaluate on the full held-out test split:

```powershell
.\.venv\Scripts\python.exe main.py --image_dir .\data\train --ground_truth_csv .\data\ISIC_2019_Training_GroundTruth.csv --metadata_csv .\data\ISIC_2019_Training_Metadata.csv --head_epochs 1 --finetune_epochs 0 --output_dir outputs_quick
```

For the existing local setup with `venv` in the parent folder, use `..\venv\Scripts\python.exe` instead of `.\.venv\Scripts\python.exe`.

## Full baseline run

```powershell
.\.venv\Scripts\python.exe main.py --image_dir .\data\train --ground_truth_csv .\data\ISIC_2019_Training_GroundTruth.csv --metadata_csv .\data\ISIC_2019_Training_Metadata.csv
```

Defaults: batch size 32, two data-loader workers, three head-training epochs, seven fine-tuning epochs, and seed 42. Fine-tuning has early stopping based on validation macro-F1. CUDA is used when available; otherwise training uses the CPU.

Progress is printed after each complete training epoch and validation pass. CPU runs can therefore remain silent for a long time. Model weights are saved only after training completes; interrupted runs cannot currently resume.

## Outputs

The chosen output directory (`outputs` by default) contains:

- `train_split.csv`, `val_split.csv`, and `test_split.csv`
- `efficientnet_b0_isic2019.pt`
- `results_summary.csv`: accuracy, balanced accuracy, precision, recall, F1, and ROC-AUC
- `classification_report.csv` and `per_class_metrics.csv`
- `confusion_matrix.png` and `roc_curves.png`
- `preliminary_results_report.txt`

Report the actual training configuration alongside results, especially for the one-epoch run. The generated narrative uses generic fine-tuning wording even when backbone fine-tuning is skipped.

## Source layout

```text
main.py              Command-line entry point
src/data_prep.py     CSV preparation and lesion-group splits
src/dataset.py       Image loading and transforms
src/model.py         EfficientNet-B0 and class weights
src/train.py         Training, validation, and prediction
src/evaluate.py      Test metrics, plots, and reports
```

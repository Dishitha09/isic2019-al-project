"""
Evaluation: produces every metric the rubric's "Preliminary results" item
asks for -- accuracy, precision, recall, F1-score, and AUC via the ROC curve
-- plus supporting plots and CSVs for the deck.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize

from src.train import predict
from src.data_prep import CLASSES


def evaluate_and_save(model, test_loader, output_dir, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred, y_prob = predict(model, test_loader, device)

    # 1) Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    # 2-4) Precision, Recall, F1 -- macro (minority-class visibility) and
    # weighted (overall performance).
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # 5) AUC via ROC curve -- one-vs-rest since this is an 8-class problem.
    y_true_bin = label_binarize(y_true, classes=np.arange(len(CLASSES)))
    macro_auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
    weighted_auc = roc_auc_score(y_true_bin, y_prob, average="weighted", multi_class="ovr")

    report_dict = classification_report(
        y_true, y_pred, labels=np.arange(len(CLASSES)),
        target_names=CLASSES, output_dict=True, zero_division=0
    )
    pd.DataFrame(report_dict).T.to_csv(output_dir / "classification_report.csv")

    per_class_rows = []
    roc_data = {}
    for i, class_name in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        class_auc = auc(fpr, tpr)
        roc_data[class_name] = (fpr, tpr, class_auc)

        cls_report = report_dict[class_name]
        per_class_rows.append({
            "class": class_name,
            "precision": cls_report["precision"],
            "recall": cls_report["recall"],
            "f1_score": cls_report["f1-score"],
            "support": int(cls_report["support"]),
            "auc_ovr": class_auc
        })

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(output_dir / "per_class_metrics.csv", index=False)

    summary = pd.DataFrame([{
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "macro_roc_auc_ovr": macro_auc,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "weighted_roc_auc_ovr": weighted_auc,
    }])
    summary.to_csv(output_dir / "results_summary.csv", index=False)

    # ROC plot -- one curve per class (one-vs-rest).
    plt.figure(figsize=(9, 7))
    for class_name in CLASSES:
        fpr, tpr, class_auc = roc_data[class_name]
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={class_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", label="No-skill")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ISIC 2019 One-vs-Rest ROC Curves (Macro AUC={macro_auc:.3f})")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=200)
    plt.close()

    # Confusion matrix -- motivates the class-aware stopping criterion later.
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASSES)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    ax.set_title("ISIC 2019 Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    report_text = f"""
PRELIMINARY RESULTS

A transfer-learning baseline was evaluated on the ISIC 2019 skin-lesion
classification task using a lesion-level train/validation/test split to
prevent images from the same lesion from appearing in more than one
partition. Images with no confirmed diagnosis (UNK) were excluded rather
than folded into another class. An ImageNet-pretrained EfficientNet-B0
network was fine-tuned for eight-class classification, with class-weighted
cross-entropy to address the dataset's class imbalance.

On the held-out test set, the model achieved a classification accuracy of
{accuracy:.4f} ({accuracy*100:.2f}%) and a balanced accuracy of
{balanced_accuracy:.4f}. Macro precision was {macro_p:.4f}, macro recall was
{macro_r:.4f}, and macro F1-score was {macro_f1:.4f}. Using a one-vs-rest
multiclass ROC evaluation, the macro-average AUC was {macro_auc:.4f}. The
corresponding weighted precision, recall and F1-score were {weighted_p:.4f},
{weighted_r:.4f}, and {weighted_f1:.4f}, with a weighted ROC-AUC of
{weighted_auc:.4f}.

Because the dataset is strongly class-imbalanced, macro-averaged metrics and
per-class recall/F1 are emphasized over accuracy alone. Per-class results are
in per_class_metrics.csv; ROC curves and confusion matrix are saved as images.

These results are the preliminary transfer-learning classification baseline
that the later active-learning pipeline (cold-start init + class-aware
stopping criterion) will aim to match with far fewer labeled samples.
""".strip()

    with open(output_dir / "preliminary_results_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + "=" * 70)
    print("PRELIMINARY TEST RESULTS")
    print("=" * 70)
    print(f"Accuracy               : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Balanced Accuracy      : {balanced_accuracy:.4f}")
    print(f"Macro Precision        : {macro_p:.4f}")
    print(f"Macro Recall           : {macro_r:.4f}")
    print(f"Macro F1-score         : {macro_f1:.4f}")
    print(f"Macro ROC-AUC (OvR)    : {macro_auc:.4f}")
    print(f"Weighted Precision     : {weighted_p:.4f}")
    print(f"Weighted Recall        : {weighted_r:.4f}")
    print(f"Weighted F1-score      : {weighted_f1:.4f}")
    print(f"Weighted ROC-AUC (OvR) : {weighted_auc:.4f}")
    print("\nPer-class metrics:")
    print(per_class_df.to_string(index=False))
    print(f"\nSaved outputs to: {output_dir.resolve()}")

    return summary, per_class_df

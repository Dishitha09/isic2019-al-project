"""
Entry point for the ISIC 2019 preliminary classification baseline.

Run from the project root, e.g.:
    python main.py --image_dir data/ISIC_2019_Training_Input \
                    --ground_truth_csv data/ISIC_2019_Training_GroundTruth.csv \
                    --metadata_csv data/ISIC_2019_Training_Metadata.csv
"""

import random
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_prep import prepare_dataframe, lesion_level_split, CLASSES
from src.dataset import ISICDataset, make_transforms
from src.model import build_model, make_class_weights
from src.train import fit_model
from src.evaluate import evaluate_and_save


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = prepare_dataframe(args.ground_truth_csv, args.metadata_csv, args.image_dir)
    train_df, val_df, test_df = lesion_level_split(df, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)

    print(f"Total images (after UNK removal): {len(df)}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("\nTrain IMAGE-level class distribution:")
    print(train_df["label"].value_counts().reindex(CLASSES).fillna(0).astype(int))

    train_tfms, eval_tfms = make_transforms()
    train_ds = ISICDataset(train_df, train_tfms)
    val_ds = ISICDataset(val_df, eval_tfms)
    test_ds = ISICDataset(test_df, eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    class_weights, counts = make_class_weights(train_df, num_classes=len(CLASSES))
    print("\nClass weights:")
    for c, w, n in zip(CLASSES, class_weights.numpy(), counts):
        print(f"{c:5s}: count={int(n):5d}, weight={w:.4f}")

    model = build_model(num_classes=len(CLASSES)).to(device)
    model = fit_model(model, train_loader, val_loader, class_weights, device,
                       head_epochs=args.head_epochs, finetune_epochs=args.finetune_epochs)

    torch.save(model.state_dict(), output_dir / "efficientnet_b0_isic2019.pt")
    evaluate_and_save(model, test_loader, output_dir, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--ground_truth_csv", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--head_epochs", type=int, default=3)
    parser.add_argument("--finetune_epochs", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

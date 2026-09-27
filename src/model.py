"""Transfer-learning backbone and class-imbalance weighting."""

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_model(num_classes=8):
    # ImageNet-pretrained EfficientNet-B0 -- the transfer-learning backbone
    # for this baseline (your cold-start sub-problem builds on this later).
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def make_class_weights(train_df, num_classes=8):
    # Inverse-sqrt class weighting: rarer classes get a higher loss weight
    # so the majority class (NV) doesn't dominate training. Inverse-sqrt is
    # gentler than plain inverse-frequency, avoiding over-weighting the
    # rarest classes to the point of destabilizing training.
    counts = train_df["label_idx"].value_counts().sort_index()
    counts = counts.reindex(range(num_classes), fill_value=1).values.astype(np.float32)

    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32), counts

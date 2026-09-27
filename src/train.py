"""Two-phase transfer-learning training loop."""

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)

        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    # Macro-F1 (not accuracy) tracks epoch quality, since accuracy alone
    # can look fine while minority classes are being ignored.
    macro_f1 = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )[2]
    return epoch_loss, macro_f1


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y_true, y_prob = [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        y_true.extend(labels.numpy())
        y_prob.append(probs.cpu().numpy())

    y_true = np.asarray(y_true)
    y_prob = np.concatenate(y_prob, axis=0)
    y_pred = y_prob.argmax(axis=1)
    return y_true, y_pred, y_prob


def val_macro_f1(model, loader, device):
    y_true, y_pred, _ = predict(model, loader, device)
    return precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )[2]


def fit_model(model, train_loader, val_loader, class_weights, device,
              head_epochs=3, finetune_epochs=7):
    """Phase 1: freeze backbone, train only the new classifier head.
    Phase 2: unfreeze the last few backbone blocks, fine-tune at a lower
    learning rate, with early stopping on validation macro-F1."""
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0

    for p in model.features.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )

    print("\nPHASE 1: classifier head")
    for epoch in range(head_epochs):
        loss, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vf1 = val_macro_f1(model, val_loader, device)
        print(f"Head epoch {epoch+1}/{head_epochs} | loss={loss:.4f} | "
              f"train macro-F1={train_f1:.4f} | val macro-F1={vf1:.4f}")
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    for p in model.features.parameters():
        p.requires_grad = False
    for block in list(model.features.children())[-3:]:
        for p in block.parameters():
            p.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    patience = 4
    bad_epochs = 0

    print("\nPHASE 2: fine-tuning")
    for epoch in range(finetune_epochs):
        loss, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vf1 = val_macro_f1(model, val_loader, device)
        scheduler.step(vf1)

        print(f"Fine-tune epoch {epoch+1}/{finetune_epochs} | loss={loss:.4f} | "
              f"train macro-F1={train_f1:.4f} | val macro-F1={vf1:.4f}")

        if vf1 > best_f1:
            best_f1 = vf1
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print("Early stopping on validation macro-F1.")
            break

    model.load_state_dict(best_state)
    return model

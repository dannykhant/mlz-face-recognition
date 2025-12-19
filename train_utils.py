import os

from PIL import Image
from tqdm import tqdm

import torch
from torch import nn, optim
from torchvision import models
from torch.utils.data import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_identity_counts(root_dir):
    identity_counts = {}

    for identity in os.listdir(root_dir):
        identity_path = os.path.join(root_dir, identity)
        if not os.path.isdir(identity_path):
            continue

        num_images = len([
            f for f in os.listdir(identity_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        identity_counts[identity] = num_images

    return identity_counts


def filter_identities(identity_counts, min_images=10):
    return {
        identity: count
        for identity, count in identity_counts.items()
        if count >= min_images
    }


class FilteredImageFolder(Dataset):
    def __init__(self, root, allowed_identities, transform=None):
        self.root = root
        self.transform = transform
        self.allowed_identities = sorted(allowed_identities)

        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.allowed_identities)
        }

        self.samples = []

        for cls_name in self.allowed_identities:
            cls_dir = os.path.join(root, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(cls_dir, fname)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    

class FaceRecognition(nn.Module):
  def __init__(self, embedding_dim=512, droprate=0.2, num_classes=10):
    super().__init__()

    self.backbone = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

    in_features = self.backbone.classifier[0].in_features
    self.backbone.classifier = nn.Identity() # type: ignore

    self.embedding = nn.Sequential(
        nn.Linear(in_features, embedding_dim), # type: ignore
        nn.BatchNorm1d(embedding_dim)
    )

    self.classifier = nn.Sequential(
        nn.Dropout(droprate),
        nn.Linear(embedding_dim, num_classes)
    )

  def forward(self, x):
    features = self.backbone(x)
    embeddings = self.embedding(features)
    logits = self.classifier(embeddings)

    return logits, embeddings


def build_model(embedding_dim=512, droprate=0.2, num_classes=10):
  model = FaceRecognition(embedding_dim,
                          droprate,
                          num_classes=num_classes)
  model.to(device)
  print(f"model is on {device}")

  return model


def configure_backbone(model, unfreeze_blocks=0):
  if unfreeze_blocks > 0:
    for p in model.backbone.features[-unfreeze_blocks:].parameters():
      p.requires_grad = True
  else:
    for p in model.backbone.parameters():
      p.requires_grad = False


def train_one_epoch(train_dl, model, optimizer, loss_fn):
  model.train()
  running_loss, total_preds, correct_preds = 0., 0, 0

  for i, (inputs, labels) in tqdm(enumerate(train_dl), desc="Training"):
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs, embeddings = model(inputs)

    loss = loss_fn(outputs, labels)
    loss.backward()

    optimizer.step()

    running_loss += loss.item()
    total_preds += labels.size(0)
    correct_preds += outputs.argmax(1).eq(labels).sum().item()

  epoch_loss = running_loss / len(train_dl)
  epoch_accuracy = correct_preds / total_preds

  return epoch_loss, epoch_accuracy


def validate(val_dl, model, loss_fn):
  model.eval()
  running_loss, total_preds, correct_preds = 0., 0, 0

  with torch.no_grad():
    for i, (inputs, labels) in tqdm(enumerate(val_dl), desc="Validating"):
      inputs, labels = inputs.to(device), labels.to(device)
      outputs, embeddings = model(inputs)
      loss = loss_fn(outputs, labels)

      running_loss += loss.item()
      total_preds += labels.size(0)
      correct_preds += outputs.argmax(1).eq(labels).sum().item()

  epoch_loss = running_loss / len(val_dl)
  epoch_accuracy = correct_preds / total_preds

  return epoch_loss, epoch_accuracy

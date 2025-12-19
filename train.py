import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

from train_utils import (
    get_identity_counts,
    filter_identities,
    FilteredImageFolder,
    build_model,
    configure_backbone,
    train_one_epoch,
    validate
)


def prepare_data(data_dir="data/lfw_funneled"):
    print("Preparing data...")

    identity_counts = get_identity_counts(data_dir)

    filtered_identities = filter_identities(identity_counts, min_images=20)

    dataset = FilteredImageFolder(
        root=data_dir,
        allowed_identities=filtered_identities
    )

    return dataset


def validation_framework(dataset):
    print("Setting up data transforms and splits...")

    input_size = 224

    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.02
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3)],
            p=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_ds.dataset.transform = train_transform # type: ignore
    val_ds.dataset.transform = val_transform # type: ignore

    return train_ds, val_ds, test_ds


def get_data_loaders(train_ds, val_ds, test_ds):
    print("Creating data loaders...")

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=32,
        shuffle=True
    )

    val_dl = DataLoader(
        dataset=val_ds,
        batch_size=32,
        shuffle=False
    )

    return train_dl, val_dl


def train_model():
    print("Starting training process...")

    dataset = prepare_data("data/lfw_funneled")
    train_ds, val_ds, test_ds = validation_framework(dataset)
    train_labels = set(dataset.samples[i][1] for i in train_ds.indices)
    train_dl, val_dl = get_data_loaders(train_ds, val_ds, test_ds)

    epochs = 1
    best_val_accuracy = 0.
    checkpoint_path = "mobilenetv3_face_recog_v1_{epoch:02d}_{val_accuracy:.3f}.pth"

    embedding_dim = 128
    unfreeze_blocks = 3
    learning_rate = 0.001

    model = build_model(embedding_dim=embedding_dim, num_classes=len(train_labels))

    configure_backbone(model, unfreeze_blocks=unfreeze_blocks)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()),
                        lr=learning_rate)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1}")

        train_result = train_one_epoch(train_dl, model, optimizer, loss_fn)
        print(f"Train | Loss: {train_result[0]} | Accuracy: {train_result[1]}")

        val_result = validate(val_dl, model, loss_fn)
        print(f"Val | Loss: {val_result[0]} | Accuracy: {val_result[1]}")

        if val_result[1] > best_val_accuracy:
            best_val_accuracy = val_result[1]
            checkpoint = checkpoint_path.format(epoch=epoch + 1,
                                                val_accuracy=val_result[1])
            torch.save(model.state_dict(), checkpoint)
            print(f"Saved best model with val_accuracy: {val_result[1]}")



if __name__ == "__main__":
    train_model()

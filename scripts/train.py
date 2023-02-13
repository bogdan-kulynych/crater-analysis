import torch
import numpy as np

import torchvision

from torchgeo import models

# from torchgeo.datasets import NonGeoClassificationDataset
# from torchgeo.trainers import ClassificationTask
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning import Trainer

from src.sat_trainer import ClassificationTask


def compute_data_stats(data_path, transform, seed=0):
    unnormalized_image_data = torchvision.datasets.ImageFolder(
        root=data_path, transform=transform
    )
    # Normalize data using full data stats. It's a bit of a leakage.
    initial_loader = DataLoader(
        unnormalized_image_data, batch_size=len(unnormalized_image_data), shuffle=False
    )

    images, labels = next(iter(initial_loader))

    # shape of images = [b,c,w,h]
    mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
    return mean, std


class TorchgeoFmtDataset(torchvision.datasets.ImageFolder):
    def __getitem__(self, *args, **kwargs):
        image, label = super().__getitem__(*args, **kwargs)
        return {"image": image, "label": label}


def train(
    model_name="resnet18",
    weights=None,
    batch_size=64,
    base_size=256,
    crop_size=224,
    learning_rate=1e-4,
    learning_rate_schedule_patience=5,
    weight_decay=0,
    optimizer="SGD",
    num_layers_to_finetune=3,
    dro="up",
    normalize=True,
):
    train_data_path = "data/train/"
    test_data_path = "data/test/"

    train_transforms = [
        transforms.Resize(base_size),
        transforms.RandomCrop(crop_size),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.ToTensor(),
    ]
    test_transforms = [
        transforms.Resize(crop_size),
        transforms.ToTensor(),
    ]

    if normalize:
        train_mean, train_std = compute_data_stats(
            train_data_path,
            transforms.Compose(test_transforms),
        )
        train_transforms.append(transforms.Normalize(train_mean, train_std))
        test_transforms.append(transforms.Normalize(train_mean, train_std))
    train_data = TorchgeoFmtDataset(
        root=train_data_path, transform=transforms.Compose(train_transforms)
    )
    test_data = TorchgeoFmtDataset(
        root=test_data_path, transform=transforms.Compose(test_transforms)
    )

    if dro == None or dro == "none":
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
        )

    # Upsampling the minority class.
    elif dro == "up":
        labels = []
        for i in range(len(train_data)):
            item = train_data[i]
            labels.append(item["label"])
        labels = np.array(labels)
        sample_weights = np.ones_like(labels)
        pos_label = 1
        neg_label = 0
        pos_prop = (labels == pos_label).mean()
        neg_prop = (labels == neg_label).mean()
        for i in range(len(labels)):
            if labels[i] == pos_label:
                sample_weights[i] /= pos_prop
            elif labels[i] == neg_label:
                sample_weights[i] /= neg_prop

        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights,
            num_samples=batch_size,
            replacement=False,
        )
        train_loader = DataLoader(
            train_data,
            sampler=sampler,
            pin_memory=True,
            num_workers=8,
        )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
    )

    if model_name == "resnet18":
        if weights is None:
            weights = "ResNet18_Weights.SENTINEL2_RGB_SECO"

    elif model_name == "resnet50":
        if weights is None:
            weights = "ResNet50_Weights.SENTINEL2_RGB_SECO"

    elif model_name == "vit_small_patch16_224":
        if weights is None:
            weights = "ViTSmall16_Weights.SENTINEL2_ALL_SECO"

    task = ClassificationTask(
        model=model_name,
        weights=weights,
        loss="ce",
        in_channels=3,
        num_classes=2,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_schedule_patience=learning_rate_schedule_patience,
        num_layers_to_finetune=num_layers_to_finetune,
        optimizer=optimizer,
        weight_decay=weight_decay,
    )
    task.config_model()
    trainer = Trainer(gpus=1)
    trainer.fit(task, train_loader, test_loader)


import fire

if __name__ == "__main__":
    fire.Fire(train)

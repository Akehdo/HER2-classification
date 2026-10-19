"""Общие архитектуры для обучения и инференса без загрузки весов при импорте."""

import torch
from torch import nn
from torchvision import models


MODELS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights),
    "resnet50": (models.resnet50, models.ResNet50_Weights),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights),
}


def create_model(name, weights, num_classes, weights_file=None):
    model = MODELS[name][0](weights=None if weights_file is not None else weights)
    if weights_file is not None:
        model.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True))
    if name == "efficientnet_b0":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

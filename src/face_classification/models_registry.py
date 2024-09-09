import torch
from torchvision.models import mobilenet_v2, resnet18, resnet50, resnet152
from torchvision.models.resnet import ResNet50_Weights


def get_resnet(backbone: str, output_classes: int) -> torch.nn.Module:
    if backbone == "resnet_18":
        model = resnet18(weights=ResNet50_Weights.IMAGENET1K_V1)
    elif backbone == "resnet_50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    elif backbone == "resnet_152":
        model = resnet152(weights=ResNet50_Weights.IMAGENET1K_V1)
    elif backbone == "mobilenet_v2":
        # model = torch.hub.load("pytorch/vision:v0.9.0", "mobilenet_v2", pretrained=True)
        model = mobilenet_v2(pretrained=True)
        print(model)
    else:
        raise ValueError(f"Architecture param passed as input could not be understood: {backbone}")
    if backbone == "mobilenet_v2":
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, output_classes)
    else:
        model.fc = torch.nn.Linear(model.fc.in_features, output_classes)
    return model


CLASSIFICATION_MODELS = {
    "resnet": get_resnet,
}

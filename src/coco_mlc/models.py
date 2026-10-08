"""Torchvision model factory for multi-label classification.

Reference: https://docs.pytorch.org/vision/main/models.html

Every model returns raw logits (no sigmoid inside the network): the losses
expect logits, and the sigmoid is applied explicitly at evaluation and
prediction time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torchvision import models as tvm

from .config import NUM_CLASSES


@dataclass(frozen=True)
class ModelInfo:
    """Metadata taken from the torchvision weights table."""

    name: str
    builder: str
    weights: str
    params_m: float
    gflops: float
    imagenet_acc1: float
    note: str = ""


# Candidates kept for the comparison. GFLOPS are given at 224x224 and
# directly drive training time, which matters on CPU.
MODEL_REGISTRY: dict[str, ModelInfo] = {
    "mobilenet_v3_small": ModelInfo(
        "mobilenet_v3_small", "mobilenet_v3_small", "MobileNet_V3_Small_Weights",
        2.5, 0.06, 67.67, "cheapest, useful to check the pipeline"),
    "shufflenet_v2_x1_0": ModelInfo(
        "shufflenet_v2_x1_0", "shufflenet_v2_x1_0", "ShuffleNet_V2_X1_0_Weights",
        2.3, 0.14, 69.36, ""),
    "mobilenet_v3_large": ModelInfo(
        "mobilenet_v3_large", "mobilenet_v3_large", "MobileNet_V3_Large_Weights",
        5.5, 0.22, 75.27, "best cost/accuracy trade-off on CPU"),
    "efficientnet_b0": ModelInfo(
        "efficientnet_b0", "efficientnet_b0", "EfficientNet_B0_Weights",
        5.3, 0.39, 77.69, "next best trade-off just above MobileNet"),
    "efficientnet_b1": ModelInfo(
        "efficientnet_b1", "efficientnet_b1", "EfficientNet_B1_Weights",
        7.8, 0.69, 79.84, "IMAGENET1K_V2 weights"),
    "efficientnet_b3": ModelInfo(
        "efficientnet_b3", "efficientnet_b3", "EfficientNet_B3_Weights",
        12.2, 1.83, 82.01, "better accuracy at about the cost of a ResNet18"),
    "efficientnet_b4": ModelInfo(
        "efficientnet_b4", "efficientnet_b4", "EfficientNet_B4_Weights",
        19.3, 4.39, 83.38, ""),
    "resnet18": ModelInfo(
        "resnet18", "resnet18", "ResNet18_Weights",
        11.7, 1.81, 69.76, "course baseline"),
    "vgg16": ModelInfo(
        "vgg16", "vgg16", "VGG16_Weights",
        138.4, 15.47, 71.59, "course architecture; expensive, not the best trade-off"),
    "resnet50": ModelInfo(
        "resnet50", "resnet50", "ResNet50_Weights",
        25.6, 4.09, 80.86, "IMAGENET1K_V2 weights (improved training recipe)"),
    "convnext_tiny": ModelInfo(
        "convnext_tiny", "convnext_tiny", "ConvNeXt_Tiny_Weights",
        28.6, 4.46, 82.52, "modern architecture, needs a GPU"),
    "swin_t": ModelInfo(
        "swin_t", "swin_t", "Swin_T_Weights",
        28.3, 4.49, 81.47, "hierarchical transformer, needs a GPU"),
    "maxvit_t": ModelInfo(
        "maxvit_t", "maxvit_t", "MaxVit_T_Weights",
        30.9, 5.56, 83.70, "best accuracy in this cost band"),
    "efficientnet_v2_s": ModelInfo(
        "efficientnet_v2_s", "efficientnet_v2_s", "EfficientNet_V2_S_Weights",
        21.5, 8.37, 84.23, "highest ImageNet accuracy among the candidates"),
}


def _last_linear(model: nn.Module) -> tuple[str, nn.Linear]:
    """Qualified name and module of the last linear layer.

    ``named_modules()`` walks the graph in definition order, so the last
    ``nn.Linear`` is the classification head for every family used here
    (``fc`` for ResNet, ``classifier`` for MobileNet / EfficientNet / ConvNeXt,
    ``heads.head`` for ViT, ``head`` for Swin).
    """
    found: tuple[str, nn.Linear] | None = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            found = (name, module)
    if found is None:
        raise RuntimeError(
            f"No nn.Linear layer found in {type(model).__name__}: "
            "this model has a non-linear head (for example SqueezeNet) and is not supported."
        )
    return found


def _set_module(model: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def _load_weights(info: ModelInfo, pretrained: bool):
    if not pretrained:
        return None
    return getattr(tvm, info.weights).DEFAULT


def build_model(
    name: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.0,
    prior=None,
) -> nn.Module:
    """Pretrained model whose head is replaced by ``num_classes`` outputs."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model: {name!r} (available: {sorted(MODEL_REGISTRY)})")
    info = MODEL_REGISTRY[name]
    model = getattr(tvm, info.builder)(weights=_load_weights(info, pretrained))

    head_name, head = _last_linear(model)
    in_features = head.in_features

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    classifier = nn.Linear(in_features, num_classes)
    if prior is not None:
        init_prior_bias(classifier, prior)
    new_head: nn.Module = classifier
    if dropout > 0:
        new_head = nn.Sequential(nn.Dropout(dropout), classifier)
    _set_module(model, head_name, new_head)
    return model


def build_feature_extractor(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Frozen backbone whose head is replaced by ``nn.Identity``.

    The model then returns the feature vector that used to feed the ImageNet
    classification layer. That is the basis of the precompute strategy: one
    forward pass over the dataset is enough to train as many heads as we want,
    at a very small extra cost.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model: {name!r} (available: {sorted(MODEL_REGISTRY)})")
    info = MODEL_REGISTRY[name]
    model = getattr(tvm, info.builder)(weights=_load_weights(info, pretrained))

    head_name, head = _last_linear(model)
    feature_dim = head.in_features
    _set_module(model, head_name, nn.Identity())

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, feature_dim


def init_prior_bias(layer: nn.Linear, prior) -> None:
    """Set the output bias to the log-odds of each class prevalence.

    With a zero bias, the network starts from probabilities of 0.5 while the
    average prevalence is 3.7%: the first epochs only push the biases down, and
    the initial loss is 0.69 instead of 0.15. Starting from the prior (as
    suggested in the focal loss paper) skips that useless phase.
    """
    p = torch.as_tensor(np.asarray(prior), dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
    with torch.no_grad():
        layer.bias.copy_(torch.log(p / (1 - p)))


class LinearHead(nn.Module):
    """Multi-label logistic regression on precomputed features."""

    def __init__(
        self,
        in_features: int,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.0,
        prior=None,
    ):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if prior is not None:
            init_prior_bias(self.classifier, prior)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(x))


class MLPHead(nn.Module):
    """Head with one hidden layer, to capture interactions between features."""

    def __init__(
        self,
        in_features: int,
        num_classes: int = NUM_CLASSES,
        hidden: int = 1024,
        dropout: float = 0.3,
        prior=None,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden, num_classes)
        if prior is not None:
            init_prior_bias(self.classifier, prior)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.trunk(x))


def build_head(kind: str, in_features: int, num_classes: int = NUM_CLASSES, **kwargs) -> nn.Module:
    kind = kind.lower()
    if kind == "linear":
        return LinearHead(in_features, num_classes, **kwargs)
    if kind == "mlp":
        return MLPHead(in_features, num_classes, **kwargs)
    raise ValueError(f"unknown head: {kind!r} (available: linear, mlp)")


def classifier_in_features(model: nn.Module) -> int:
    """Size of the vector that feeds the classification head."""
    return _last_linear(model)[1].in_features


def load_compatible_weights(model: nn.Module, source: nn.Module) -> list[str]:
    """Copy tensors from ``source`` into ``model`` when the name or shape matches.

    A ``Linear`` head and the same head wrapped in ``Sequential(Dropout, Linear)``
    do not share keys (``fc.weight`` versus ``fc.1.weight``). The second form is
    recovered by dropping the ``.1`` index of the dropout.
    """
    src = source.state_dict()
    dst = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    copied: list[str] = []
    for key, tensor in dst.items():
        candidates = [key]
        if key.endswith(".1.weight"):
            candidates.append(key[: -len(".1.weight")] + ".weight")
        elif key.endswith(".1.bias"):
            candidates.append(key[: -len(".1.bias")] + ".bias")
        for cand in candidates:
            if cand in src and tuple(src[cand].shape) == tuple(tensor.shape):
                mapped[key] = src[cand]
                copied.append(key)
                break
    model.load_state_dict({**dst, **mapped}, strict=False)
    return copied


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

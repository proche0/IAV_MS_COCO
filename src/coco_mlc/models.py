"""Fabrique de modeles torchvision pour la classification multi-label.

Reference : https://docs.pytorch.org/vision/main/models.html

Tous les modeles renvoient des logits bruts (aucune sigmoide dans le reseau) :
les fonctions de cout utilisees attendent des logits, et la sigmoide est
appliquee explicitement au moment de l'evaluation et de la prediction.
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
    """Metadonnees issues du tableau des poids de la doc torchvision."""

    name: str
    builder: str
    weights: str
    params_m: float
    gflops: float
    imagenet_acc1: float
    note: str = ""


# Candidats retenus pour l'etude comparative. Les GFLOPS sont donnes a 224x224
# et conditionnent directement le temps d'entrainement, critique en CPU.
MODEL_REGISTRY: dict[str, ModelInfo] = {
    "mobilenet_v3_small": ModelInfo(
        "mobilenet_v3_small", "mobilenet_v3_small", "MobileNet_V3_Small_Weights",
        2.5, 0.06, 67.67, "le moins couteux, utile pour valider le pipeline"),
    "shufflenet_v2_x1_0": ModelInfo(
        "shufflenet_v2_x1_0", "shufflenet_v2_x1_0", "ShuffleNet_V2_X1_0_Weights",
        2.3, 0.14, 69.36, ""),
    "mobilenet_v3_large": ModelInfo(
        "mobilenet_v3_large", "mobilenet_v3_large", "MobileNet_V3_Large_Weights",
        5.5, 0.22, 75.27, "meilleur rapport cout/precision en CPU"),
    "efficientnet_b0": ModelInfo(
        "efficientnet_b0", "efficientnet_b0", "EfficientNet_B0_Weights",
        5.3, 0.39, 77.69, "meilleur rapport juste au-dessus de MobileNet"),
    "efficientnet_b1": ModelInfo(
        "efficientnet_b1", "efficientnet_b1", "EfficientNet_B1_Weights",
        7.8, 0.69, 79.84, "poids IMAGENET1K_V2"),
    "efficientnet_b3": ModelInfo(
        "efficientnet_b3", "efficientnet_b3", "EfficientNet_B3_Weights",
        12.2, 1.83, 82.01, "meilleure accuracy au cout d'un ResNet18"),
    "efficientnet_b4": ModelInfo(
        "efficientnet_b4", "efficientnet_b4", "EfficientNet_B4_Weights",
        19.3, 4.39, 83.38, ""),
    "resnet18": ModelInfo(
        "resnet18", "resnet18", "ResNet18_Weights",
        11.7, 1.81, 69.76, "baseline de reference du cours"),
    "resnet50": ModelInfo(
        "resnet50", "resnet50", "ResNet50_Weights",
        25.6, 4.09, 80.86, "poids IMAGENET1K_V2 (recette d'entrainement amelioree)"),
    "convnext_tiny": ModelInfo(
        "convnext_tiny", "convnext_tiny", "ConvNeXt_Tiny_Weights",
        28.6, 4.46, 82.52, "architecture moderne, necessite un GPU"),
    "swin_t": ModelInfo(
        "swin_t", "swin_t", "Swin_T_Weights",
        28.3, 4.49, 81.47, "transformer hierarchique, necessite un GPU"),
    "maxvit_t": ModelInfo(
        "maxvit_t", "maxvit_t", "MaxVit_T_Weights",
        30.9, 5.56, 83.70, "meilleure accuracy de cette bande de cout"),
    "efficientnet_v2_s": ModelInfo(
        "efficientnet_v2_s", "efficientnet_v2_s", "EfficientNet_V2_S_Weights",
        21.5, 8.37, 84.23, "le plus precis sur ImageNet parmi les candidats"),
}


def _last_linear(model: nn.Module) -> tuple[str, nn.Linear]:
    """Nom qualifie et module de la derniere couche lineaire du reseau.

    ``named_modules()`` parcourt le graphe dans l'ordre de definition, donc la
    derniere ``nn.Linear`` est la tete de classification pour toutes les
    familles utilisees ici (``fc`` pour ResNet, ``classifier`` pour MobileNet /
    EfficientNet / ConvNeXt, ``heads.head`` pour ViT, ``head`` pour Swin).
    """
    found: tuple[str, nn.Linear] | None = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            found = (name, module)
    if found is None:
        raise RuntimeError(
            f"Aucune couche nn.Linear trouvee dans {type(model).__name__} : "
            "ce modele a une tete non lineaire (ex. SqueezeNet) et n'est pas supporte."
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
    """Modele pre-entraine dont la tete est remplacee par ``num_classes`` sorties."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"modele inconnu : {name!r} (disponibles : {sorted(MODEL_REGISTRY)})")
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
    """Backbone gele dont la tete est remplacee par ``nn.Identity``.

    Le modele renvoie alors le vecteur de features qui alimentait la couche de
    classification ImageNet. C'est la base de la strategie de pre-calcul : un
    seul passage forward sur le dataset suffit ensuite a entrainer autant de
    tetes que l'on veut, pour un cout negligeable.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"modele inconnu : {name!r} (disponibles : {sorted(MODEL_REGISTRY)})")
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
    """Initialise le biais de sortie a la log-cote de la prevalence de chaque classe.

    Avec un biais nul, le reseau part de probabilites de 0,5 alors que la
    prevalence moyenne est de 3,7 % : les premieres epoques ne servent qu'a faire
    descendre les biais, et la perte initiale vaut 0,69 au lieu de 0,15. Partir
    du prior (recommandation de l'article sur la focal loss) supprime cette
    phase inutile.
    """
    p = torch.as_tensor(np.asarray(prior), dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
    with torch.no_grad():
        layer.bias.copy_(torch.log(p / (1 - p)))


class LinearHead(nn.Module):
    """Regression logistique multi-label sur features pre-calculees."""

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
    """Tete a une couche cachee, pour capter les interactions entre features."""

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
    raise ValueError(f"tete inconnue : {kind!r} (disponibles : linear, mlp)")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

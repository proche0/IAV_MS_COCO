# Modèles et poids pré-entraînés torchvision

Référence : <https://docs.pytorch.org/vision/main/models.html>

Le sujet impose une **stratégie de transfer learning**. Ce document explique
l'API torchvision utilisée, comment nous adaptons un réseau ImageNet à notre
problème à 80 sorties, et comment les candidats de l'étude comparative ont été
choisis.

---

## 1. L'API des poids

torchvision expose les poids pré-entraînés par des énumérations, une par
architecture :

```python
from torchvision.models import resnet50, ResNet50_Weights

weights = ResNet50_Weights.DEFAULT        # meilleurs poids disponibles
model = resnet50(weights=weights)
preprocess = weights.transforms()         # prétraitement associé aux poids
categories = weights.meta["categories"]   # les 1000 classes ImageNet
```

Trois points importants :

- `DEFAULT` pointe vers les meilleurs poids disponibles, pas forcément
  `IMAGENET1K_V1`. Pour ResNet50, c'est `IMAGENET1K_V2`, qui gagne 4,7 points de
  top-1 (80,86 % contre 76,13 %) **à architecture et coût de calcul identiques**,
  uniquement grâce à une meilleure recette d'entraînement. C'est un gain gratuit.
- `weights.transforms()` donne le prétraitement exact associé aux poids. La doc
  insiste : « Using the correct preprocessing method is critical. »
- Les poids sont téléchargés une fois dans `~/.cache/torch/hub/checkpoints`
  (surchargeable par `TORCH_HOME`).

### Pourquoi nous n'utilisons pas `weights.transforms()` directement

Le prétraitement par défaut est `Resize(256) → CenterCrop(224)`. Il suppose des
images de résolution native supérieure à 256 px. Nos images ont un **grand côté
de 224 px exactement** : `Resize(256)` sur le petit côté agrandirait l'image
(une image 224×149 deviendrait 385×256), puis `CenterCrop(224)` recadrerait en
supprimant les bords gauche et droit.

En multi-label, supprimer les bords supprime des objets dont la classe reste dans
la cible : on crée du bruit d'étiquetage. Nous conservons donc la normalisation
ImageNet (indispensable) mais gérons la géométrie nous-mêmes, avec le mode `pad`
par défaut — voir [partie 3](03-outils-et-evaluation.md#30-le-dataset-réel).

---

## 2. Adapter un réseau ImageNet à 80 sorties multi-label

Deux modifications, et une seule qui touche l'architecture.

### Remplacer la tête de classification

Le réseau pré-entraîné sort 1 000 logits ImageNet. Il faut une couche à 80
sorties. L'emplacement de cette couche diffère selon la famille :

| Architecture | Attribut de la tête | Dimension des features |
| --- | --- | --- |
| resnet18 | `fc` | 512 |
| resnet50 | `fc` | 2048 |
| shufflenet_v2_x1_0 | `fc` | 1024 |
| mobilenet_v3_small | `classifier.3` | 1024 |
| mobilenet_v3_large | `classifier.3` | 1280 |
| efficientnet_b0 | `classifier.1` | 1280 |
| efficientnet_v2_s | `classifier.1` | 1280 |
| convnext_tiny | `classifier.2` | 768 |
| swin_t | `head` | 768 |

Plutôt que de coder ces cas un par un, `models.build_model` localise
**la dernière `nn.Linear` du graphe** : `named_modules()` parcourt les modules
dans l'ordre de définition, donc la dernière couche linéaire est la tête de
classification pour toutes ces familles. Ajouter une architecture au registre ne
demande aucun code spécifique.

### Ne pas ajouter de sigmoïde

Le réseau renvoie des **logits bruts**. La sigmoïde est appliquée :

- implicitement par `BCEWithLogitsLoss`, qui est numériquement plus stable que
  `Sigmoid` suivi de `BCELoss` ;
- explicitement à l'évaluation et à la prédiction, avant le seuillage.

C'est la source de confusion la plus fréquente sur ce projet : seuiller à 0,5
des sorties qui sont des logits revient à seuiller les probabilités à 0,62.

### Le mode « extraction de features »

`models.build_feature_extractor` remplace la dernière `nn.Linear` par
`nn.Identity` au lieu d'une nouvelle couche. Le réseau renvoie alors le vecteur
de features qui alimentait la tête ImageNet — 512 valeurs pour ResNet18, 1 280
pour MobileNetV3-Large.

Comme ces features ne dépendent pas de l'apprentissage (le backbone est gelé),
elles peuvent être calculées une fois et mises en cache. C'est la base de notre
stratégie en l'absence de GPU, décrite en
[partie 4](04-programme-entrainement.md#40-pourquoi-deux-programmes).

Deux stratégies de transfer learning sont donc comparables :

| Stratégie | Ce qui est appris | Coût | Quand |
| --- | --- | --- | --- |
| Extraction de features | la tête seulement | 1 forward + quelques secondes par expérience | CPU, étude comparative large |
| Fine-tuning | tout le réseau | plusieurs époques complètes | GPU, meilleure performance finale |

---

## 3. Les candidats retenus

Extrait du tableau des poids de la doc torchvision (précision top-1 ImageNet,
GFLOPS à 224×224), complété par le débit mesuré sur notre CPU
([`scripts/benchmark_speed.py`](../scripts/benchmark_speed.py)) :

| Modèle | Params | GFLOPS | top-1 | Forward CPU | Entraînement CPU | Rôle |
| --- | --- | --- | --- | --- | --- | --- |
| mobilenet_v3_small | 2,5 M | 0,06 | 67,7 % | 271 img/s | 71 img/s | valider le pipeline rapidement |
| shufflenet_v2_x1_0 | 2,3 M | 0,14 | 69,4 % | — | — | alternative très légère |
| **mobilenet_v3_large** | 5,5 M | 0,22 | 75,3 % | 78 img/s | 23 img/s | meilleur rapport coût/précision en CPU |
| efficientnet_b0 | 5,3 M | 0,39 | 77,7 % | — | — | plus précis pour un coût proche |
| **resnet18** | 11,7 M | 1,81 | 69,8 % | 50 img/s | 17 img/s | baseline de référence du cours |
| **resnet50** | 25,6 M | 4,09 | **80,9 %** | 17 img/s | 5,6 img/s | référence GPU, poids `IMAGENET1K_V2` |
| convnext_tiny | 28,6 M | 4,46 | 82,5 % | — | — | architecture convolutive moderne |
| swin_t | 28,3 M | 4,49 | 81,5 % | — | — | transformer hiérarchique |
| efficientnet_v2_s | 21,5 M | 8,37 | 84,2 % | — | — | le plus précis, le plus coûteux |

Registre dans [`models.py`](../src/coco_mlc/models.py) (`MODEL_REGISTRY`), avec
les métadonnées pour les tableaux comparatifs.

### Logique de sélection

**Le coût de calcul n'est pas une considération secondaire ici, c'est la
contrainte principale.** À budget fixe, mieux vaut comparer proprement trois
architectures et calibrer les seuils que lancer un seul EfficientNetV2-S à moitié
entraîné.

- **ResNet18** est la baseline : c'est l'architecture du TP, elle sert de point
  de comparaison avec le travail déjà fait sur CIFAR-10.
- **MobileNetV3-Large** est intéressant au-delà de sa légèreté : il est 8 fois
  moins coûteux que ResNet18 (0,22 contre 1,81 GFLOPS) **tout en étant plus
  précis sur ImageNet** (75,3 % contre 69,8 %). Son vecteur de features est
  aussi plus riche (1 280 contre 512).
- **ResNet50 avec les poids V2** est le candidat sérieux pour la soumission
  finale : 80,9 % de top-1, et des features de 2 048 dimensions qui se prêtent
  bien à une tête linéaire.
- **ConvNeXt-Tiny** et **Swin-T** permettent de comparer une convolution moderne
  et un transformer à coût de calcul quasi identique (4,46 contre 4,49 GFLOPS),
  comparaison intéressante pour le rapport. Réservés au GPU.

### Observation attendue sur la précision ImageNet

La corrélation entre top-1 ImageNet et performance sur notre tâche n'est pas
parfaite. ResNet18 a une top-1 inférieure à MobileNetV3-Large mais des features
de dimension plus faible et une architecture différente. C'est précisément ce que
l'étude comparative doit mesurer plutôt que supposer — et la mAP (indépendante du
seuil) est la bonne grandeur pour cette comparaison, car elle isole la qualité du
classement de l'effet de la calibration.

---

## 4. Autres familles de modèles torchvision

La page documente aussi des modèles pour d'autres tâches, hors sujet ici mais
utiles à situer :

- **détection d'objets** (Faster R-CNN, RetinaNet, FCOS, SSD) — prédisent des
  boîtes englobantes. Les poids `COCO_V1` sont entraînés sur MS COCO complet :
  les utiliser reviendrait à exploiter une autre distribution de MS COCO, ce que
  le sujet **interdit explicitement** ;
- **segmentation sémantique** (FCN, DeepLabV3, LRASPP) ;
- **classification vidéo**, **flot optique**, **ré-identification de personnes**.

Seules les architectures de **classification d'images avec poids ImageNet-1K**
sont utilisées ici.

### Le point d'attention sur les licences

La doc torchvision précise que les poids pré-entraînés peuvent avoir leurs
propres licences, dérivées du dataset d'entraînement. Les poids ImageNet-1K sont
destinés à la recherche et à l'enseignement, ce qui couvre cet usage.

---

## 5. Ce que le sujet interdit

> Il va sans dire qu'il est **interdit** d'utiliser une autre distribution de la
> base MS COCO pour l'entraînement, par exemple le dataset Torchvision.

Concrètement :

- interdit : `torchvision.datasets.CocoDetection`, les poids de détection
  `COCO_V1`, ou tout téléchargement d'annotations MS COCO ;
- autorisé : les poids **ImageNet-1K** de classification, qui constituent
  précisément le transfer learning demandé.

Le code n'importe aucun dataset torchvision ; les seules données lues sont celles
du répertoire `ms-coco` fourni.

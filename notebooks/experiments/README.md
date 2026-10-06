# Notebooks d'expérimentation

Un notebook par architecture du registre, générés par
[`../build_experiment_notebooks.py`](../build_experiment_notebooks.py).

| Notebook | Modèle |
| --- | --- |
| [exp_mobilenet_v3_small.ipynb](exp_mobilenet_v3_small.ipynb) | MobileNetV3-Small |
| [exp_shufflenet_v2_x1_0.ipynb](exp_shufflenet_v2_x1_0.ipynb) | ShuffleNet V2 x1.0 |
| [exp_mobilenet_v3_large.ipynb](exp_mobilenet_v3_large.ipynb) | MobileNetV3-Large |
| [exp_efficientnet_b0.ipynb](exp_efficientnet_b0.ipynb) | EfficientNet-B0 |
| [exp_efficientnet_b1.ipynb](exp_efficientnet_b1.ipynb) | EfficientNet-B1 |
| [exp_efficientnet_b3.ipynb](exp_efficientnet_b3.ipynb) | EfficientNet-B3 |
| [exp_efficientnet_b4.ipynb](exp_efficientnet_b4.ipynb) | EfficientNet-B4 |
| [exp_resnet18.ipynb](exp_resnet18.ipynb) | ResNet18 |
| [exp_vgg16.ipynb](exp_vgg16.ipynb) | VGG16 |
| [exp_resnet50.ipynb](exp_resnet50.ipynb) | ResNet50 |
| [exp_convnext_tiny.ipynb](exp_convnext_tiny.ipynb) | ConvNeXt-Tiny |
| [exp_swin_t.ipynb](exp_swin_t.ipynb) | Swin-T |
| [exp_maxvit_t.ipynb](exp_maxvit_t.ipynb) | MaxViT-T |
| [exp_efficientnet_v2_s.ipynb](exp_efficientnet_v2_s.ipynb) | EfficientNetV2-S |

Chaque notebook enchaîne le split stratifié 70/15/15, l'augmentation, une
baseline à extracteur gelé, puis le fine-tuning (augmentation renforcée,
dropout, weight decay). Le F1 du serveur est le critère. Les courbes tracent
l'erreur `100 × (1 − F1)`.

`FULL_TRAIN` vaut `False` : 512 images et une époque, pour vérifier le
pipeline. Le passer à `True` en tête de notebook lance l'expérience complète.
Les essais courts n'écrasent pas le cache du split complet.

Le test officiel n'a pas d'étiquettes. Le « test local » est une part des
images annotées, lue une seule fois en fin de notebook. Le diagnostic
compare ensuite les erreurs d'entraînement, de validation et de test.

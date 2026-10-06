# IAV — Challenge de classification multi-label MS COCO

Repository du challenge MS COCO en IAV pour Tayeb et Paul.

Classification **multi-label** de 80 classes MS COCO : chaque image peut
contenir plusieurs catégories d'objets. Énoncé complet dans
[`sujet.ipynb`](sujet.ipynb).

| | |
| --- | --- |
| Données | 65 000 images d'entraînement annotées, 4 952 images de test |
| Sorties | 80 classes, vecteur multi-hot |
| Métrique | F1 avec précision et rappel **pondérés par l'inverse de la fréquence** |
| Soumission | [leaderboard](https://www.creatis.insa-lyon.fr/kechichian/ms-coco-classif-leaderboard.html) au format JSON |

---

## Le point qui détermine tout

La métrique du serveur pondère chaque classe par `1/fréquence`. Résultat :

- `hair drier` (102 images) pèse **13,6 %** du score, `toaster` (117) **11,9 %** ;
- les **10 classes les plus rares pèsent 43,8 %** du score ;
- `person` (35 494 images) pèse **0,039 %**, soit 350 fois moins que `hair drier`.

Optimiser l'accuracy ou le micro-F1 mène donc à un mauvais score. Le levier
principal est le **rappel sur les classes rares**, obtenu par des fonctions de
coût adaptées au déséquilibre et par une **calibration des seuils classe par
classe**. Analyse détaillée dans
[`docs/03-outils-et-evaluation.md`](docs/03-outils-et-evaluation.md).

---

## Installation

```bash
# torch et torchvision doivent venir du meme build (cpu ou cu12x)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Le dataset n'est pas dans le dépôt. Il est attendu dans `../ms-coco` par rapport
à la racine du dépôt, ou à l'emplacement indiqué par `MSCOCO_ROOT` :

```bash
export MSCOCO_ROOT=/chemin/vers/ms-coco
```

Structure attendue : `images/train/`, `images/test/`, `labels/train/`.

---

## Démarrage rapide

```bash
# 0. Vérifier la métrique et les données
python3 tests/test_metrics_parity.py
python3 scripts/explore_dataset.py

# 1. Pré-calculer les features d'un backbone gelé (une seule fois, ~16 min en CPU)
python3 scripts/cache_features.py --model mobilenet_v3_large

# 2. Baseline, puis étude comparative des coûts et des têtes (secondes par run)
python3 scripts/train_head.py --model mobilenet_v3_large
python3 scripts/train_head.py --model mobilenet_v3_large --sweep

# 3. Calibrer les seuils par classe sur la validation
python3 scripts/tune_thresholds.py --checkpoint outputs/head_mobilenet_v3_large_mlp_bce.pth --save

# 4. Produire le JSON de soumission
python3 scripts/predict.py --checkpoint outputs/head_mobilenet_v3_large_mlp_bce.pth --submit-copy
```

Meilleur modèle actuel (CPU, backbone gelé) : **MobileNetV3-Large + MLP + BCE**,
seuils par classe, F1 validation **0,6145**. Détail dans [`PROGRESS.md`](PROGRESS.md).

Fine-tuning complet (à réserver au GPU, voir
[`notebooks/colab_finetune.ipynb`](notebooks/colab_finetune.ipynb)) :

```bash
python3 scripts/train.py --model resnet50 --epochs 10 --loss asl --amp --tensorboard
```

---

## Organisation du dépôt

```
src/coco_mlc/          bibliothèque réutilisable
├── config.py          chemins, 80 classes, hyperparamètres par défaut
├── data.py            datasets, transformations, découpage stratifié
├── models.py          fabrique torchvision, têtes de classification
├── losses.py          BCE pondérée, focal loss, asymmetric loss
├── metrics.py         métrique du serveur reproduite, macro/micro F1, mAP
├── features.py        extraction et cache des features
├── engine.py          boucles train/eval, checkpoints, Tensorboard
├── thresholds.py      calibration des seuils de décision
├── inference.py       reconstruction d'un modèle depuis un checkpoint
└── utils.py           graines, registre d'expériences

scripts/               programmes exécutables
├── explore_dataset.py statistiques et figures
├── benchmark_speed.py débit réel des backbones sur la machine
├── cache_features.py  un forward sur tout le dataset, backbone gelé
├── train_head.py      entraînement de têtes sur features (étude comparative)
├── train.py           fine-tuning complet (partie 4 du sujet)
├── tune_thresholds.py calibration des seuils
└── predict.py         JSON de soumission (partie 5 du sujet)

docs/                  explication des parties 3, 4, 5 du sujet
notebooks/             analyse des résultats, fine-tuning Colab
tests/                 parité de la métrique avec le code du sujet
outputs/               résultats, checkpoints, figures (hors git)
features/              caches de features (hors git)
```

---

## Documentation

| Document | Contenu |
| --- | --- |
| [`docs/03-outils-et-evaluation.md`](docs/03-outils-et-evaluation.md) | les `Dataset`, les boucles, **la métrique du serveur décortiquée**, le découpage train/validation |
| [`docs/04-programme-entrainement.md`](docs/04-programme-entrainement.md) | le squelette du sujet mis en correspondance avec le code, et pourquoi deux programmes d'entraînement |
| [`docs/05-programme-soumission.md`](docs/05-programme-soumission.md) | format JSON, pièges de l'inférence, vérifications automatiques |
| [`docs/06-modeles-preentraines.md`](docs/06-modeles-preentraines.md) | API des poids torchvision, adaptation à 80 sorties, candidats de l'étude |
| [`PROGRESS.md`](PROGRESS.md) | journal de bord, expériences, scores leaderboard |

---

## Règles du challenge à ne pas oublier

- Une soumission portant le même nom de groupe **écrase** la précédente :
  consulter le classement avant d'envoyer, et ne pas créer d'entrée en double.
- Il est **interdit** d'utiliser une autre distribution de MS COCO, en
  particulier le dataset torchvision ou les poids de détection `COCO_V1`. Seuls
  les poids **ImageNet** sont utilisés.
- Le test ne sert jamais à choisir un hyperparamètre ou un seuil : tout se décide
  sur la validation.

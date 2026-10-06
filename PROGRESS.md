# Journal de bord — Challenge MS COCO multi-label

Groupe : Tayeb et Paul. Dernière mise à jour : 6 octobre 2026.

---

## État d'avancement

| Étape | État |
| --- | --- |
| Environnement (torch + torchvision CUDA 12.8, consignes dans `requirements.txt`) | installation à lancer |
| Bibliothèque `src/coco_mlc/` (données, modèles, coûts, métriques, moteur, seuils) | fait |
| Métrique du serveur reproduite et vérifiée contre le sujet | fait |
| Exploration du dataset, statistiques et figures | fait |
| Benchmark des débits CPU (ancienne machine) | fait |
| Benchmark des débits RTX 4060 Ti | à faire — commande dans la section matériel |
| Cache de features MobileNetV3-Large | fait |
| Cache de features ResNet18 | fait (32 min 33 s, 33,3 img/s) |
| Cache de features ResNet50 | non fait ; fine-tuning local à la place |
| Étude comparative têtes × fonctions de coût | fait (7 configs MobileNet + 3 ResNet18) |
| Comparaison de backbones gelés (MobileNet vs ResNet18) | fait |
| Calibration des seuils par classe | fait |
| JSON de soumission produit et vérifié | fait |
| Pipeline de fine-tuning (`scripts/train.py`) | fait (smoke test 512 images OK) |
| Fine-tuning complet en local (`scripts/train.py --amp`) | à faire |
| Soumission au leaderboard | à faire — JSON prêt |
| Notebook d'analyse | fait |
| Documentation parties 3, 4, 5, 6 | fait |

---

## Le constat qui a structuré le travail

La métrique du serveur pondère la précision et le rappel de chaque classe par
`1/fréquence`. Sur les 65 000 images d'entraînement :

- `hair drier` (102 positifs) pèse **13,6 %** du score, `toaster` (117) **11,9 %** ;
- les **10 classes les plus rares pèsent 43,8 %** du score ;
- les **40 classes les plus fréquentes réunies ne pèsent que 22,0 %** ;
- `person` (35 494 positifs) pèse **0,039 %**, soit 350 fois moins que `hair drier`.

Toute la stratégie en découle : viser le rappel sur les classes rares, et
calibrer les seuils classe par classe. Détail dans
[`docs/03-outils-et-evaluation.md`](docs/03-outils-et-evaluation.md).

Repère : prédire **toutes** les classes sur **toutes** les images donne
F1 = 0,0337. C'est le plancher à battre largement.

---

## Contrainte matérielle et stratégie retenue

Machine de travail : NVIDIA RTX 4060 Ti, Intel i9-13900KF, 32 Go de RAM.
L'extraction de features et le fine-tuning complet tournent tous les deux en
local. Le fine-tuning se lance avec `scripts/train.py` et `--amp` (précision
mixte). Les poids ImageNet (`Weights.DEFAULT`) sont téléchargés
automatiquement au premier entraînement.

Débits de la RTX 4060 Ti, à remplir après :

```bash
python scripts/benchmark_speed.py --models mobilenet_v3_small mobilenet_v3_large efficientnet_b0 efficientnet_b1 efficientnet_b3 resnet18 efficientnet_b4 resnet50 convnext_tiny swin_t maxvit_t efficientnet_v2_s --batch-size 32 --batches 8
```

Le CSV est écrit dans `outputs/benchmark_speed.csv`. Le script mesure la
précision classique, pas `--amp`, et n'a pas besoin des poids pré-entraînés.

| Backbone | Forward | Entraînement | Cache 70 k | 1 époque (52 k) |
| --- | --- | --- | --- | --- |
| mobilenet_v3_small | | | | |
| mobilenet_v3_large | | | | |
| efficientnet_b0 | | | | |
| efficientnet_b1 | | | | |
| efficientnet_b3 | | | | |
| resnet18 | | | | |
| efficientnet_b4 | | | | |
| resnet50 | | | | |
| convnext_tiny | | | | |
| swin_t | | | | |
| maxvit_t | | | | |
| efficientnet_v2_s | | | | |

L'étude comparative des têtes reste sur des features mises en cache : un seul
passage forward, puis quelques secondes par expérience. Le fine-tuning
complet, plus performant, se fait sur cette machine. Justification dans
[`docs/04-programme-entrainement.md`](docs/04-programme-entrainement.md).

Temps réel de l'extraction sur l'ancienne machine (Intel i5, sans GPU) :

- MobileNetV3-Large : **15 min 40 s** train (69,2 img/s) + 1 min 10 s test, cache 179 Mo.
- ResNet18 : **32 min 33 s** train (33,3 img/s) + 1 min 47 s test, cache 77 Mo.

ResNet50 s'entraîne en local :

```bash
python scripts/train.py --model resnet50 --epochs 10 --loss asl --amp
```

---

## Expériences

Toutes sur MobileNetV3-Large gelé, features de dimension 1280, 224 px en mode
`pad`, découpage stratifié 80/20 graine 42 (52 047 train / 12 953 validation),
AdamW lr 1e-3, 40 époques, sélection sur le F1 serveur au meilleur seuil global.
Registre complet : `outputs/experiments.csv`.

### Ablation : initialisation du biais de sortie

Première observation marquante. Avec un biais de sortie nul, le réseau part de
probabilités de 0,5 alors que la prévalence moyenne est de 3,7 % : les premières
époques ne servent qu'à faire descendre les biais.

| Initialisation du biais | F1 validation | mAP | Meilleure époque |
| --- | --- | --- | --- |
| zéro | 0,4574 | 0,6241 | 40 (encore en progression) |
| log-cote de la prévalence | **0,5317** | 0,6413 | 5 |

Gain de **+16 %** de F1, et convergence en 5 époques au lieu de plus de 40.
Appliqué partout ensuite.

### Étude comparative : têtes × fonctions de coût

Une seule variable change à la fois par rapport à la baseline (tête linéaire +
BCE). Scores au meilleur seuil **global** — la calibration par classe vient
après et s'applique à toutes.

| Tête | Coût | F1 | Précision | Rappel | mAP | Meilleure époque | Durée |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | bce | **0,5614** | 0,5539 | 0,5692 | **0,6549** | 3 | 4 min 12 |
| MLP | bce_pos_weight | 0,5512 | 0,5516 | 0,5509 | 0,6287 | 22 | 4 min 14 |
| MLP | asl | 0,5407 | 0,5508 | 0,5309 | 0,6405 | 13 | 4 min 24 |
| linéaire | bce | 0,5317 | 0,6681 | 0,4416 | 0,6413 | 5 | 36 s |
| linéaire | bce_pos_weight | 0,5176 | 0,6128 | 0,4480 | 0,5815 | 2 | 36 s |
| linéaire | focal | 0,4829 | 0,5042 | 0,4633 | 0,6160 | 7 | 41 s |
| linéaire | asl | 0,4804 | 0,5576 | 0,4220 | 0,6054 | 11 | 45 s |

Trois enseignements :

1. **La tête MLP apporte +0,03 de F1 et +0,014 de mAP** sur la tête linéaire. Le
   gain est réel mais modeste pour 7 fois plus de temps de calcul, ce qui suggère
   que les features du backbone gelé sont déjà presque linéairement séparables
   pour cette tâche.
2. **Les coûts conçus pour le déséquilibre font moins bien que la BCE simple**,
   ce qui est contre-intuitif. Explication la plus probable : focal et ASL
   agissent surtout sur la *calibration* des sorties, or la calibration est
   justement traitée séparément et plus directement par l'optimisation des
   seuils. Leurs hyperparamètres (γ, clip, plafond de `pos_weight`) n'ont pas
   été réglés — c'est la première piste à explorer.
3. **Le surapprentissage est très rapide** : les meilleures époques sont 3 et 5.
   Une baisse du learning rate ou une régularisation plus forte est à tester.

### Comparaison de backbones gelés (même protocole)

Même découpage, même coût BCE, même calibration au meilleur seuil global :

| Backbone | Tête | Dropout | F1 | mAP | Features |
| --- | --- | --- | --- | --- | --- |
| **MobileNetV3-Large** | MLP | 0,3 | **0,5614** | **0,6549** | 1280 |
| MobileNetV3-Large | linéaire | 0 | 0,5317 | 0,6413 | 1280 |
| ResNet18 | MLP | 0,3 | 0,4757 | 0,5932 | 512 |
| ResNet18 | MLP | 0 | 0,4759 | 0,5851 | 512 |
| ResNet18 | linéaire | 0 | 0,4623 | 0,5642 | 512 |

MobileNetV3-Large gèle **mieux** que ResNet18, cohérent avec ImageNet (75,3 % vs
69,8 % top-1) et avec un vecteur plus riche (1280 vs 512), malgré 8 fois moins
de GFLOPS. Après calibration par classe : MobileNet **0,6145**, ResNet18
**0,5358**. C'est donc MobileNetV3-Large + MLP + BCE qui est soumis. Le
fine-tuning GPU devra battre ce 0,6145 pour justifier un changement de modèle.

---

## Calibration des seuils

Sur le meilleur modèle (MLP + BCE), calibration par montée de coordonnées sur
**exactement** la métrique du serveur, **uniquement sur la validation** :

| Stratégie de seuillage | F1 validation | Gain |
| --- | --- | --- |
| seuil 0,5 (défaut) | 0,4479 | référence |
| meilleur seuil unique (0,14) | 0,5614 | **+25,4 %** |
| un seuil par classe | **0,6145** | **+37,2 %** |

Les seuils calibrés vont de 0,02 à 0,86, médiane 0,24. Les classes les plus
lourdes reçoivent les seuils les plus bas, exactement comme attendu :
`toaster` 0,02, `hot dog` 0,02, `donut` 0,06, `hair drier` 0,10.

**La calibration sur-apprend-elle la validation ?** Contrôle : seuils réglés sur
une moitié de la validation, évalués sur l'autre moitié.

| Seuils | F1 sur la moitié non utilisée |
| --- | --- |
| 0,5 | 0,4482 |
| unique, calibré | 0,5782 |
| par classe, calibrés | **0,6128** |

Les seuils par classe généralisent : 0,6128 contre 0,6145 en apprentissage, soit
une perte de 0,3 %. La calibration n'est donc pas un artefact.

À retenir : **la calibration des seuils vaut plus que tout ce que l'étude
comparative d'architectures a apporté** (+37 % contre +6 %). C'est la
conséquence directe de la pondération de la métrique.

---

## Soumission

Fichier produit : `submissions/predictions_head_mobilenet_v3_large_mlp_bce.json`

| Contrôle | Résultat |
| --- | --- |
| Entrées | 4 952 / 4 952 attendues |
| Identifiants au format 12 chiffres | conformes |
| Indices entiers dans [0, 79] | conformes |
| Listes vides | aucune (28 images rattrapées par le garde-fou, 0,6 %) |
| Classes jamais prédites | 0 |
| Classes par image | moyenne 7,09 (référence train : 2,93) |

La moyenne de 7,09 classes par image est **voulue** : les seuils optimisés pour
la métrique pondérée échangent de la précision contre du rappel. Vérification de
cohérence encourageante — l'image `000000000285` est prédite `[21]` (`bear`),
exactement la valeur donnée en exemple dans le sujet.

### Historique des soumissions au leaderboard

| Date | Modèle | F1 validation | F1 leaderboard | Note |
| --- | --- | --- | --- | --- |
| — | MobileNetV3-Large gelé + MLP + BCE, seuils par classe | 0,6145 | *non soumis* | prête à envoyer |

**Rappel : une soumission portant le même nom de groupe écrase la précédente.**
Consulter le classement avant d'envoyer, ne pas créer d'entrée en double, et
reporter chaque score dans ce tableau.

---

## Prochaines étapes, par ordre de rentabilité estimée

1. **Mesurer les débits sur la RTX 4060 Ti**, puis **fine-tuner en local**
   (`scripts/train.py --amp`). C'est le gain le plus important attendu : le
   backbone gelé plafonne à une mAP de 0,65, et adapter les features à MS COCO
   devrait dépasser cela nettement. Candidats : ResNet50 (poids
   `IMAGENET1K_V2`, 80,9 % top-1), EfficientNet-B3/B4, ConvNeXt-Tiny, Swin-T,
   MaxVit-T, EfficientNetV2-S. La commande de benchmark est dans la section
   matérielle.
2. **Régler les hyperparamètres d'ASL et de la focal loss** (γ⁻, clip, plafond de
   `pos_weight`), puisque leur sous-performance actuelle vient probablement de
   valeurs non ajustées.
3. **Corriger le surapprentissage précoce** : learning rate plus bas, dropout,
   weight decay plus fort, arrêt anticipé plus serré.
4. **Augmentation de données au test (TTA)** par miroir horizontal : moyenne des
   scores de l'image et de son symétrique. Coût faible, gain habituel de 1 à 2 %.
5. **Ensemble** de plusieurs backbones par moyenne des probabilités, puis
   recalibration des seuils sur l'ensemble.
6. **Comparer les modes géométriques** `pad`, `squash` et `crop`. Un seul cache
   de features par mode suffit à trancher.

---

## Journal

### 6 octobre 2026

- Changement de machine : NVIDIA RTX 4060 Ti, Intel i9-13900KF, 32 Go de RAM.
  Les consignes d'installation passent à PyTorch CUDA 12.8 (`cu128`, repli
  `cu124`). Le fine-tuning se lance en local avec `scripts/train.py --amp`.
  Le notebook Colab est retiré.
- Short-list de 12 backbones pour le benchmark de débit. Quatre entrent dans
  le registre : EfficientNet-B1, EfficientNet-B3, EfficientNet-B4, MaxVit-T.

### 5 octobre 2026

- Installation : `torchvision` de PyPI est compilé pour CUDA et ne se charge pas
  avec un `torch` en version `+cpu` (« operator torchvision::nms does not
  exist »). Il faut l'installer depuis l'index CPU de PyTorch. Noté dans
  `requirements.txt` pour ne pas y repasser.
- Le notebook `MS_COCO_classification_project.ipynb` initialement présent dans le
  dépôt ne pouvait pas fonctionner : il suppose un dossier `images/` plat et
  déduit le sous-ensemble de test comme « les images sans fichier `.cls` », alors
  que le dataset a des dossiers `images/train`, `images/test` et `labels/train`
  séparés. Remplacé par la bibliothèque `src/coco_mlc/`.
- Mise en évidence de la pondération `1/fréquence` de la métrique. C'est le point
  qui a réorienté toute la stratégie.
- Parité de notre métrique avec celle du sujet vérifiée automatiquement
  (`tests/test_metrics_parity.py`, écart < 1e-5). Deux divisions par zéro du code
  fourni corrigées : classe sans positif en validation, et précision ou rappel
  agrégé nul aux premières époques.
- Découpage stratifié : erreur de support de 0,10 % contre 3,45 % pour un tirage
  aléatoire. Important parce que les classes rares dominent le score.
- Le déterminisme est confirmé : un sweep interrompu puis relancé a produit des
  F1 identiques au quatrième chiffre.
- Cache ResNet18 terminé (32 min 33 s). À protocole identique, MobileNetV3-Large
  MLP (F1 0,5614 / mAP 0,6549) bat ResNet18 MLP (F1 0,4757 / mAP 0,5932). La
  hiérarchie ImageNet se transpose, amplifiée par un vecteur plus large
  (1280 vs 512).
- Pipeline `scripts/train.py` validé sur 512 images (1 époque) : checkpoint,
  sélection sur F1 calibré, seuils par classe. Prêt pour Colab.
- JSON de soumission régénéré : 4 952/4 952, 0 classe oubliée, 28 images
  rattrapées, moyenne 7,09 classes/image.

# Partie 3 — Outils fournis et évaluation

Cette partie du sujet décrit les briques de code mises à disposition et le
mécanisme d'évaluation. Ce document explique ce que fait chaque brique, comment
nous l'avons reprise dans `src/coco_mlc/`, et surtout **comment la métrique du
serveur est construite** — point qui détermine toute notre stratégie.

---

## 3.0 Le dataset réel

Le sujet annonce deux sous-répertoires `images` et `labels`. Le dataset fourni
est en réalité organisé ainsi :

```
ms-coco/
├── images/
│   ├── train/   65 000 fichiers .jpg
│   └── test/     4 952 fichiers .jpg
└── labels/
    └── train/   65 000 fichiers .cls
```

Vérifications effectuées par [`scripts/explore_dataset.py`](../scripts/explore_dataset.py) :

| Grandeur | Valeur |
| --- | --- |
| Images d'entraînement annotées | 65 000 |
| Images de test | 4 952 |
| Annotations totales | 190 192 |
| Classes par image | moyenne 2,93, médiane 2, min 1, max 18 |
| Recouvrement train/test | aucun |
| Dimensions | grand côté toujours 224 px, petit côté de 59 à 224 px |

Un fichier `.cls` contient un indice de classe par ligne. L'indice correspond à
la position dans le tuple `CLASSES` défini dans
[`src/coco_mlc/config.py`](../src/coco_mlc/config.py).

**Conséquence du format des images.** Les images ont été redimensionnées en
conservant leur ratio, avec le grand côté à 224. Elles n'ont donc pas toutes la
même forme (24 % sont en portrait, les ratios vont de 0,37 à 3,80). Trois
options de mise en forme sont implémentées dans `build_transforms` :

- `pad` (notre défaut) : complète l'image en carré par des bordures noires puis
  redimensionne. Aucun contenu n'est perdu, le ratio est préservé.
- `squash` : `Resize((224, 224))` direct. Simple, mais déforme l'image.
- `crop` : redimensionne le petit côté à 224 puis recadre au centre. Agrandit
  l'image et **supprime les bords**, où se trouvent souvent des objets à
  prédire — risqué en multi-label.

---

## 3.1 Les `Dataset` personnalisés

Le sujet fournit `COCOTrainImageDataset` et `COCOTestImageDataset`. Nous les
avons reprises dans [`src/coco_mlc/data.py`](../src/coco_mlc/data.py) avec la
même logique :

- la liste des exemples d'entraînement est dérivée des fichiers `.cls` triés, et
  le chemin de l'image est déduit du nom de fichier ;
- la cible est un vecteur multi-hot de 80 valeurs. Le sujet l'obtient par
  `torch.zeros(80).scatter_(0, torch.tensor(labels), value=1)`, nous écrivons
  l'équivalent plus direct `labels[class_ids] = 1.0` ;
- le dataset de test retourne `(image, identifiant)`, l'identifiant étant le nom
  de fichier sans extension — exactement la clé attendue dans le JSON de
  soumission.

Deux ajouts :

- `return_id=True` sur le dataset d'entraînement, pour pouvoir tracer une
  prédiction jusqu'à son image ;
- `TransformSubset`, qui restreint un dataset à une liste d'indices **avec sa
  propre transformation**. C'est nécessaire parce que l'augmentation aléatoire
  doit s'appliquer au train mais pas à la validation, alors que les deux
  proviennent du même dataset de base. `torch.utils.data.random_split`, suggéré
  par le sujet, ne permet pas cette distinction.

### Lecture rapide des annotations

`load_label_matrix()` lit les 65 000 fichiers `.cls` et renvoie une matrice
`(65000, 80)` en `uint8`, sans décoder aucune image (environ 1 seconde). Elle
sert à calculer les statistiques de classes, les poids de la métrique, le
découpage stratifié et les `pos_weight` des fonctions de coût.

---

## 3.2 Les boucles d'entraînement et de validation

Les fonctions `train_loop` et `validation_loop` du sujet sont reprises dans
[`src/coco_mlc/engine.py`](../src/coco_mlc/engine.py) sous les noms
`train_one_epoch` et `evaluate`. Nous avons changé trois choses.

**1. L'évaluation accumule les sorties avant de calculer les métriques.**
`validation_loop` met à jour des compteurs au fil des mini-batches. Nous
concaténons d'abord tous les logits et toutes les cibles (`collect_outputs`),
puis déléguons le calcul à `metrics.py`. Le coût mémoire est négligeable
(13 000 × 80 flottants) et le bénéfice important : les mêmes scores servent à
calculer la métrique, à chercher les seuils optimaux et à produire le rapport
par classe, sans jamais repasser dans le réseau.

**2. Le seuillage s'applique aux probabilités, pas aux sorties brutes.**
Le code du sujet écrit :

```python
predictions = torch.where(outputs > th_multi_label, 1.0, 0.0)   # th = 0.5
```

Comparer des **logits** à 0,5 n'a pas de sens : avec `BCEWithLogitsLoss` le
réseau ne se termine pas par une sigmoïde, et le seuil neutre sur les logits est
0 (soit une probabilité de 0,5). Nous appliquons donc explicitement
`torch.sigmoid` avant de seuiller. C'est la convention retenue partout dans le
projet : **le réseau renvoie des logits, la sigmoïde est appliquée à
l'évaluation et à la prédiction**.

**3. Deux divisions par zéro corrigées** (voir section suivante).

L'argument `one_hot` de `validation_loop` est inutile ici : nos cibles sont
toujours des vecteurs multi-hot, donc le cas `one_hot=False` (qui convertit un
vecteur d'indices en matrice) ne se présente jamais.

---

## 3.3 La métrique du serveur — le point central

### Comment elle est calculée

Pour chaque classe `c`, à partir des vrais positifs `tp`, des faux positifs `fp`
et du nombre de positifs réels `total` :

```
precision[c] = tp[c] / (tp[c] + fp[c])        (0 si tp[c] == 0)
recall[c]    = tp[c] / total[c]               (0 si tp[c] == 0)
```

Puis — et c'est là que tout se joue — l'agrégation **pondère chaque classe par
l'inverse de sa fréquence** :

```
poids[c]  = (1 / total[c]) / somme(1 / total)
precision = somme(precision[c] * poids[c])
recall    = somme(recall[c] * poids[c])
F1        = 2 / (1/precision + 1/recall)
```

Ce n'est donc **ni** un macro-F1 (qui donnerait le même poids à chaque classe),
**ni** un micro-F1 (qui donnerait le même poids à chaque décision). C'est une
moyenne harmonique de deux moyennes pondérées par la rareté.

### Ce que cela implique concrètement

Poids réels calculés sur les 65 000 images d'entraînement
(`outputs/class_stats.csv`) :

| Classe | Positifs | Poids dans le score |
| --- | --- | --- |
| hair drier | 102 | **13,64 %** |
| toaster | 117 | **11,89 %** |
| parking meter | 396 | 3,51 % |
| bear | 516 | 2,70 % |
| scissors | 555 | 2,51 % |
| toothbrush | 561 | 2,48 % |
| ... | | |
| chair | 7 043 | 0,198 % |
| person | 35 494 | **0,039 %** |

- Les **2 classes les plus rares pèsent 25,5 %** du score.
- Les **10 classes les plus rares pèsent 43,8 %**.
- Les **40 classes les plus fréquentes réunies ne pèsent que 22,0 %**.
- `hair drier` pèse **350 fois plus** que `person`.

Trois conséquences qui orientent tout le projet :

1. **Optimiser l'accuracy ou le micro-F1 est contre-productif.** Un modèle qui
   excelle sur `person`, `car` et `chair` et ignore `hair drier` aura un très
   bon micro-F1 et un très mauvais score serveur.
2. **Le rappel sur les classes rares est le levier principal.** Il vaut mieux
   accepter beaucoup de faux positifs sur `toaster` que de n'en détecter aucun :
   avec `tp = 0`, précision et rappel de la classe valent 0 et on perd
   directement ~12 % du score.
3. **Un seuil unique à 0,5 est très sous-optimal.** D'où le module
   [`thresholds.py`](../src/coco_mlc/thresholds.py), qui calibre un seuil par
   classe en maximisant exactement cette métrique.

Repère utile : un modèle qui **prédit toutes les classes sur toutes les images**
obtient un rappel de 1,0, une précision de 0,0171 et donc un F1 de **0,0337**.
Tout score inférieur à cela est pire que la stratégie la plus bête possible.

### Les deux corrections apportées

Notre implémentation ([`src/coco_mlc/metrics.py`](../src/coco_mlc/metrics.py))
reproduit la formule à l'identique — c'est vérifié automatiquement par
[`tests/test_metrics_parity.py`](../tests/test_metrics_parity.py), qui compare
nos résultats à la fonction du sujet recopiée telle quelle, sur quatre
configurations aléatoires (écart < 1e-5). Deux cas dégénérés sont traités
différemment :

| Cas | Code du sujet | Notre version |
| --- | --- | --- |
| Une classe sans aucun positif dans l'ensemble évalué | `1/0 = inf` → poids `nan` sur **toutes** les classes | poids 0 pour cette classe, renormalisation sur les autres |
| `precision` ou `recall` agrégé nul | `2/(1/0 + ...)` → division par zéro | F1 = 0 |

Le second cas est fréquent aux premières époques, quand le modèle ne prédit
encore presque rien. Sans correction, l'entraînement s'arrête sur une exception.

### Métriques complémentaires suivies

La métrique du serveur seule est difficile à interpréter : un même score peut
venir d'un bon classement mal calibré ou d'un classement médiocre bien calibré.
Nous suivons donc aussi :

- **macro-F1 et micro-F1**, pour situer le score pondéré ;
- **mAP** (moyenne des average precision par classe), qui est **indépendante du
  seuil**. C'est la bonne grandeur pour comparer la qualité brute de deux
  backbones, puisque la calibration des seuils est une étape séparée ;
- l'« accuracy » du sujet (`somme(tp) / somme(positifs)`), qui est en fait le
  **rappel micro** — nous conservons son nom pour pouvoir comparer, en sachant
  ce qu'elle mesure vraiment.

---

## 3.4 Le découpage train / validation

Le sujet suggère `torch.utils.data.random_split`. Nous proposons les deux
stratégies, sélectionnables par `--split` :

- `random` : tirage aléatoire avec graine, équivalent de la suggestion du sujet ;
- `stratified` (notre défaut) : **stratification itérative multi-label**
  (Sechidis et al., 2011). L'algorithme traite les classes de la plus rare à la
  plus fréquente et répartit leurs exemples en fonction du déficit restant de
  chaque sous-ensemble.

Pourquoi : les classes rares dominent le score mais ont très peu d'exemples. Un
tirage aléatoire peut laisser 18 positifs de `hair drier` en validation là où on
en attend 20,4 — un écart de 12 % sur une classe qui pèse 13,6 % du score, donc
une métrique locale bruitée et peu fiable pour la sélection de modèle.

Mesure sur notre split (80 / 20, graine 42) :

| | Stratifié | Aléatoire |
| --- | --- | --- |
| Erreur relative totale sur les supports attendus | **0,10 %** | 3,45 % |
| Support minimal en validation | 20 | 18 |

Le découpage est **mis en cache** dans `outputs/split_*.json` : toutes les
expériences (features pré-calculées, fine-tuning, calibration des seuils)
partagent exactement les mêmes indices, sinon les comparaisons n'auraient pas de
sens.

---

## 3.5 Le logging Tensorboard

La fonction `update_graphs` du sujet est reprise dans `engine.py`, simplifiée :
une seule boucle sur les métriques au lieu d'un appel par métrique, et les
nouvelles grandeurs (mAP, macro/micro-F1) y sont intégrées. Activation par
`--tensorboard` sur `scripts/train.py`, visualisation avec :

```bash
tensorboard --logdir runs
```

---

## 3.6 Le serveur d'évaluation

- URL : https://www.creatis.insa-lyon.fr/kechichian/ms-coco-classif-leaderboard.html
- Le classement est trié par F1.
- **Une soumission portant le même nom de groupe écrase la précédente.** Il faut
  consulter le classement avant d'envoyer, ne jamais créer de doublon d'entrée
  pour le groupe, et conserver un historique privé des scores (voir
  [`PROGRESS.md`](../PROGRESS.md)).
- Le nombre de requêtes n'est pas limité, mais le test ne doit **jamais** servir
  à choisir des hyperparamètres ou des seuils : ce rôle appartient à la
  validation. Sinon, le score du leaderboard cesse d'estimer la généralisation.

---

## Correspondance sujet → code

| Élément du sujet | Dans ce dépôt |
| --- | --- |
| `COCOTrainImageDataset` | `data.COCOTrainImageDataset` |
| `COCOTestImageDataset` | `data.COCOTestImageDataset` |
| `train_loop` | `engine.train_one_epoch` |
| `validation_loop` | `engine.evaluate` + `metrics.all_metrics` |
| Métrique pondérée | `metrics.server_metrics` (parité testée) |
| `update_graphs` | `engine.update_graphs` |
| `random_split` | `data.get_split` (`random` ou `stratified`) |

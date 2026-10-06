# Partie 5 — Le programme de soumission

La partie 5 du sujet décrit le second programme, plus simple : charger le modèle
entraîné, prédire sur les 4 952 images de test et écrire le fichier JSON à
soumettre. Implémentation : [`scripts/predict.py`](../scripts/predict.py).

« Plus simple » ne veut pas dire sans risque : c'est l'étape où une erreur
silencieuse coûte un score. Ce document détaille le format attendu, les pièges,
et les vérifications automatiques que fait le script.

---

## 5.1 Le format attendu

```json
{
    "000000000139": [56, 60, 62],
    "000000000285": [21],
    "000000000632": [57, 59, 73]
}
```

Règles :

- la clé est le **nom du fichier image sans extension** — ici toujours 12
  chiffres, c'est ce que renvoie `Path(img_path).stem` dans le dataset de test ;
- la valeur est la **liste des indices de classes** prédits, entiers dans
  `[0, 79]`, correspondant aux positions dans le tuple `CLASSES` ;
- le fichier doit contenir une entrée par image de test, soit **4 952 entrées**.

Le sujet insiste : le format doit être suivi **strictement**.

---

## 5.2 Le squelette du sujet, section par section

### Imports

`json` de la bibliothèque standard, comme suggéré par le sujet.

### Hyperparamètres d'inférence — « n'oubliez pas le seuil »

Le sujet signale le seuil de probabilité multi-label comme point d'attention.
Dans notre cas, le seuil n'est pas un hyperparamètre à deviner : il est
**calibré sur la validation et stocké dans le checkpoint** par
`train_head.py` / `train.py` / `tune_thresholds.py`. `predict.py` le relit via
`inference.checkpoint_thresholds` et affiche ce qu'il utilise :

```
Seuils     : par classe, min 0.06 / median 0.24 / max 0.62
```

`--threshold 0.3` permet de forcer un seuil unique, utile pour mesurer l'écart
entre une soumission calibrée et une soumission naïve.

### Répertoires et noms de fichiers

Dérivés de `PATHS`. Le JSON va dans `outputs/predictions_<nom du checkpoint>.json`,
et `--submit-copy` en place une copie dans `submissions/`, répertoire suivi par
git pour garder la trace exacte de ce qui a été envoyé.

### Initialisation du device

`engine.pick_device()`, comme pour l'entraînement.

### Transforms, dataset et data loader

Les transformations d'inférence doivent être **identiques à celles de la
validation** : même taille, même mode géométrique, même normalisation, et aucune
augmentation. `predict.py` lit `image_size` et `resize_mode` **depuis le
checkpoint** au lieu de les redemander en ligne de commande : c'est la seule
façon de garantir qu'un modèle entraîné en `pad` à 224 px ne soit pas évalué en
`squash` à 192 px, erreur qui dégraderait les prédictions sans aucun message
d'erreur.

Le loader ne mélange pas les images (`shuffle=False`), donc l'ordre des
identifiants du dataset correspond à celui des logits accumulés.

### Chargement du modèle depuis le fichier sauvegardé

[`inference.load_model`](../src/coco_mlc/inference.py) reconstruit l'un ou
l'autre des deux types de checkpoints et renvoie dans les deux cas un
`nn.Module` qui prend des images et renvoie des logits :

- `kind="full_model"` — reconstruit l'architecture (sans télécharger les poids
  ImageNet, inutiles ici) et charge le `state_dict` ;
- `kind="feature_head"` — reconstruit le backbone gelé, puis la tête **précédée
  de la normalisation des features**. Les statistiques de centrage/réduction
  sont enregistrées dans le checkpoint et appliquées comme un `buffer` du
  modèle : les oublier produirait des prédictions incohérentes avec
  l'entraînement, sans erreur visible.

Pour un checkpoint `feature_head`, si le cache de features de test existe déjà,
`predict.py` l'utilise et évite de repasser dans le backbone — quelques secondes
au lieu de plusieurs minutes. `--no-cache` force le passage complet, ce qui
permet de vérifier que les deux chemins donnent bien le même résultat.

### Boucle de prédiction

Pour chaque mini-batch : calcul des logits, `sigmoid`, comparaison aux seuils,
puis écriture des indices retenus dans le dictionnaire de sortie.

### Écriture du JSON

`json.dump` avec indentation. Taille typique : environ 0,3 Mo.

---

## 5.3 Les pièges, et comment ils sont traités

### Logits contre probabilités

Nos réseaux renvoient des logits. Seuiller des logits à 0,5 — ce que fait
littéralement le code du sujet — reviendrait à seuiller les probabilités à 0,62,
par accident. `predict.py` applique toujours `torch.sigmoid` avant comparaison.

### Les listes vides

Avec des seuils calibrés, certaines images peuvent n'avoir **aucune** classe
au-dessus de son seuil. Une liste vide est un faux négatif garanti : elle ne peut
rien rapporter, et elle ne protège d'aucun faux positif puisque la précision est
calculée par classe. `--min-labels 1` (défaut) retient donc la classe la plus
probable dans ce cas. Le script indique combien d'images ont été rattrapées.

### Le nombre d'entrées

Si le modèle plante sur une image ou si un chemin est mal configuré, on peut
produire un fichier à 4 900 entrées qui s'envoie sans erreur et sous-estime le
score. `validate()` compare le nombre d'entrées au nombre réel d'images de test.

### La cohérence des identifiants

Les clés doivent être les noms de fichiers, pas des indices de boucle ni des
chemins complets. `validate()` vérifie le motif à 12 chiffres.

### Les types JSON

`json` ne sérialise pas les entiers NumPy. Un `np.int64` non converti lève une
exception, ou devient une chaîne selon le sérialiseur. Les indices sont
explicitement convertis en `int` Python, et `validate()` le vérifie.

---

## 5.4 Les vérifications automatiques

`predict.py` refuse d'écrire le fichier si l'une de ces conditions échoue :

| Vérification | Pourquoi |
| --- | --- |
| nombre d'entrées = nombre d'images de test | détecte une boucle incomplète ou un mauvais répertoire |
| identifiants au motif `\d{12}` | détecte des clés mal formées |
| aucune liste vide | détecte un garde-fou désactivé ou des seuils trop hauts |
| indices entiers dans `[0, 79]` | détecte un décalage d'indices ou un type NumPy |

Et il affiche des indicateurs de vraisemblance, qui ne bloquent pas mais qui
doivent être lus :

- **nombre moyen de classes par image**, à comparer aux 2,93 du train. Une
  moyenne de 0,5 signale des seuils trop sévères, une moyenne de 15 des seuils
  trop permissifs ;
- **nombre de classes jamais prédites**. Chaque classe absente des prédictions
  contribue 0 au score, et comme le poids est inversement proportionnel à la
  fréquence, oublier `hair drier` coûte à lui seul 13,6 % du score
  ([voir partie 3](03-outils-et-evaluation.md#33-la-métrique-du-serveur--le-point-central)).

Exemple de sortie :

```
  images traitees        : 4,952 (attendu 4,952)
  classes par image      : moyenne 4.21, mediane 4, min 1, max 14
  reference train        : 2.93 classes par image
  images sans prediction : 37 rattrapees par le garde-fou (0.7 %)
  classes jamais predites: 0
  format verifie : OK
```

Note : avec des seuils optimisés pour la métrique pondérée, la moyenne de classes
par image est **attendue plus haute** que 2,93. C'est le comportement voulu : on
échange de la précision contre du rappel là où le score se joue.

---

## 5.5 Utilisation

```bash
# Soumission calibrée
python3 scripts/predict.py \
    --checkpoint outputs/head_resnet18_linear_asl.pth \
    --submit-copy

# Comparaison avec un seuil naïf, pour mesurer l'apport de la calibration
python3 scripts/predict.py \
    --checkpoint outputs/head_resnet18_linear_asl.pth \
    --threshold 0.5 --output outputs/predictions_naive.json
```

---

## 5.6 Discipline de soumission

Rappels du sujet, à respecter :

- le serveur est à utiliser **avec le plus grand soin** ;
- une nouvelle soumission portant le même nom de groupe **écrase** la
  précédente : consulter le classement avant d'envoyer ;
- **ne pas créer d'entrée en double** pour le groupe ;
- n'envoyer que des fichiers JSON au format requis ;
- garder un historique privé des scores — c'est le rôle de
  [`PROGRESS.md`](../PROGRESS.md).

Et une règle méthodologique qui ne vient pas du sujet mais qui conditionne la
valeur de nos résultats : **le test ne sert jamais à choisir quoi que ce soit**.
Architecture, hyperparamètres, fonction de coût et seuils sont tous décidés sur
la validation. Le score du leaderboard n'est lu que comme une confirmation.
Chaque fois qu'on choisit sur le test, le score cesse d'estimer la
généralisation.

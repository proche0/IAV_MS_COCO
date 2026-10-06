"""Boite a outils du challenge MS COCO multi-label (80 classes).

Modules :
  config     chemins, liste des classes, hyper-parametres par defaut
  data       datasets, transformations, decoupage train/validation
  models     fabrique de modeles torchvision et de tetes de classification
  losses     BCE ponderee, focal loss, asymmetric loss
  metrics    metrique du serveur reproduite, macro/micro F1, mAP
  engine     boucles d'entrainement et d'evaluation, checkpoints, Tensorboard
  thresholds calibration des seuils de decision
  diagnostics courbes d'erreur, schema du modele, diagnostic biais/variance
"""

from .config import CLASSES, DEFAULTS, NUM_CLASSES, PATHS, SEED

__all__ = ["CLASSES", "NUM_CLASSES", "PATHS", "SEED", "DEFAULTS"]

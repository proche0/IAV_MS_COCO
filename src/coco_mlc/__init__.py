"""Toolbox for the MS COCO multi-label challenge (80 classes).

Modules:
  config      paths, class list, default hyperparameters
  data        datasets, transforms, train/validation split
  models      torchvision model factory and classification heads
  losses      weighted BCE, focal loss, asymmetric loss
  metrics     server metric, macro/micro F1, mAP
  engine      training and evaluation loops, checkpoints, TensorBoard
  thresholds  decision-threshold calibration
  diagnostics error curves, model diagram, bias/variance diagnosis
"""

from .config import CLASSES, DEFAULTS, NUM_CLASSES, PATHS, SEED

__all__ = ["CLASSES", "NUM_CLASSES", "PATHS", "SEED", "DEFAULTS"]

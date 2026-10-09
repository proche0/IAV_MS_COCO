# Experiment notebooks

## Experiment 1

Ten architectures, each trained once with the protocol in the report: 5 epochs with a frozen backbone, then 8 epochs of fine-tuning. The notebooks are in [`1/`](1/).

| Notebook | Model |
| --- | --- |
| [exp_shufflenet_v2_x1_0.ipynb](1/exp_shufflenet_v2_x1_0.ipynb) | ShuffleNet V2 x1.0 |
| [exp_mobilenet_v3_large.ipynb](1/exp_mobilenet_v3_large.ipynb) | MobileNetV3-Large |
| [exp_efficientnet_b0.ipynb](1/exp_efficientnet_b0.ipynb) | EfficientNet-B0 |
| [exp_efficientnet_b4.ipynb](1/exp_efficientnet_b4.ipynb) | EfficientNet-B4 |
| [exp_resnet18.ipynb](1/exp_resnet18.ipynb) | ResNet18 |
| [exp_resnet50.ipynb](1/exp_resnet50.ipynb) | ResNet50 |
| [exp_convnext_tiny.ipynb](1/exp_convnext_tiny.ipynb) | ConvNeXt-Tiny |
| [exp_swin_t.ipynb](1/exp_swin_t.ipynb) | Swin-T |
| [exp_maxvit_t.ipynb](1/exp_maxvit_t.ipynb) | MaxViT-T |
| [exp_efficientnet_v2_s.ipynb](1/exp_efficientnet_v2_s.ipynb) | EfficientNetV2-S |

Each notebook runs the stratified 70/15/15 split, data augmentation, a frozen-backbone baseline, then fine-tuning (stronger augmentation, dropout, weight decay). The server F1 is the score we care about. The curves plot the error `100 × (1 − F1)`. Results stay in `outputs/notebooks/<architecture>/`.

## Experiment 2

MaxViT-T and ConvNeXt-Tiny are trained again: 10 epochs with a frozen backbone, then 10 epochs of fine-tuning. Images are loaded in the main process (`NUM_WORKERS = 0`). Mixed precision is on when CUDA is available. The notebooks are in [`2/`](2/). Results go to `outputs/notebooks/2/<architecture>/`, and the submission JSON is `submissions/exp2_<architecture>.json`.

| Notebook | Model |
| --- | --- |
| [exp_maxvit_t.ipynb](2/exp_maxvit_t.ipynb) | MaxViT-T |
| [exp_convnext_tiny.ipynb](2/exp_convnext_tiny.ipynb) | ConvNeXt-Tiny |

`FULL_TRAIN` is `True` in these notebooks. Set it to `False` for a short check (512 images, one epoch). Short runs do not overwrite the cache of the full split.

The official test set has no labels. The "local test" is a slice of the labeled images, read once at the end of the notebook. The diagnosis then compares train, validation, and test errors.

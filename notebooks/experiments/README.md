# Experiment notebooks

One notebook per architecture in the model registry.

| Notebook | Model |
| --- | --- |
| [exp_mobilenet_v3_small.ipynb](exp_mobilenet_v3_small.ipynb) | MobileNetV3-Small |
| [exp_shufflenet_v2_x1_0.ipynb](exp_shufflenet_v2_x1_0.ipynb) | ShuffleNet V2 x1.0 |
| [exp_mobilenet_v3_large.ipynb](exp_mobilenet_v3_large.ipynb) | MobileNetV3-Large |
| [exp_efficientnet_b0.ipynb](exp_efficientnet_b0.ipynb) | EfficientNet-B0 |
| [exp_efficientnet_b4.ipynb](exp_efficientnet_b4.ipynb) | EfficientNet-B4 |
| [exp_resnet18.ipynb](exp_resnet18.ipynb) | ResNet18 |
| [exp_vgg16.ipynb](exp_vgg16.ipynb) | VGG16 |
| [exp_resnet50.ipynb](exp_resnet50.ipynb) | ResNet50 |
| [exp_convnext_tiny.ipynb](exp_convnext_tiny.ipynb) | ConvNeXt-Tiny |
| [exp_swin_t.ipynb](exp_swin_t.ipynb) | Swin-T |
| [exp_maxvit_t.ipynb](exp_maxvit_t.ipynb) | MaxViT-T |
| [exp_efficientnet_v2_s.ipynb](exp_efficientnet_v2_s.ipynb) | EfficientNetV2-S |

Each notebook runs the stratified 70/15/15 split, data augmentation, a frozen-backbone baseline, then fine-tuning (stronger augmentation, dropout, weight decay). The server F1 is the score we care about. The curves plot the error `100 × (1 − F1)`.

`FULL_TRAIN` is `False`: 512 images and one epoch, to check the pipeline. Set it to `True` at the top of the notebook to run the full experiment. Short runs do not overwrite the cache of the full split.

The official test set has no labels. The "local test" is a slice of the labeled images, read once at the end of the notebook. The diagnosis then compares train, validation, and test errors.

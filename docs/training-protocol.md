# Training protocol

These rules govern how the model is trained and evaluated. They keep the reported scores honest. See the [project overview](../EXPLANATION.md) for a walkthrough and the [project requirements](requirements.md) for the system built around the model.

| Item | Rule |
| --- | --- |
| Task | Predict one of 200 Tiny ImageNet classes for each 64 × 64 image. |
| Model | A CNN designed in this project and trained from scratch. |
| Metrics | Accuracy and macro F1 on the official evaluation split. |

## 1. Model objective

The architecture is written by hand, and every learned weight is trained from random initialization on Tiny ImageNet's official training data. The model predicts exactly one class label for each input. The official evaluation split supplies the two final metrics.

## 2. Dataset and split roles

Tiny ImageNet is a subset of ImageNet prepared for Stanford CS231n. It has 200 classes and 64 × 64 images. Convert every image to RGB before preprocessing because the set includes some grayscale images.

| Split | Size | Role |
| --- | ---: | --- |
| Official `train` | 100,000 images, 500 per class | Source for both the fit and internal validation portions. Keep the two portions disjoint. |
| Official `valid` (Stanford `val/`) | 10,000 images, 50 per class | Final evaluation only. Use it once, after choosing the architecture, settings, and checkpoint. |
| Official test | 10,000 images | Not used. The labels are private. |

A stratified **90/10 split of official `train`** gives 90,000 fit images and 10,000 internal validation images, with 450 and 50 images per class respectively. Internal validation is used to tune the model and choose when to stop training. Save the split seed or indices so the split can be reproduced.

## 3. Download options

### Hugging Face (preferred)

Install `datasets`, then load the training split first:

```python
from datasets import load_dataset

train_data = load_dataset("zh-plus/tiny-imagenet", split="train")
```

Load `split="valid"` into its own variable in the final evaluation section. On Kaggle Notebooks, enable Internet in the notebook settings before downloading. [Hugging Face dataset](https://huggingface.co/datasets/zh-plus/tiny-imagenet)

### Original Stanford archive

The archive is about 240 MB. After extraction, `tiny-imagenet-200/train/` contains one folder per class. The `tiny-imagenet-200/val/` folder contains the 10,000 evaluation images and `val_annotations.txt` with their labels. Use the same class-to-index mapping for training and evaluation. [Stanford CS231n archive](http://cs231n.stanford.edu/tiny-imagenet-200.zip)

## 4. Data leakage and preprocessing

**Leakage invalidates the final scores.** Split the official training data first. Fit all learned preprocessing, including normalization means and standard deviations, on the fit portion only. Apply the same normalization to internal validation and final evaluation. Apply training augmentations only to the fit portion.

The following all violate the official evaluation holdout:

- Moving or copying evaluation images into a development split.
- Training on flipped, cropped, or otherwise augmented copies of evaluation images.
- Merging official training and evaluation images and splitting them again.
- Picking a model, learning rate, epoch, or other setting based on the official evaluation score.

Keep the evaluation dataset separate from the training code. Compute final predictions after selecting the model and finishing the training run.

## 5. Allowed and excluded methods

| Allowed | Excluded |
| --- | --- |
| A framework such as PyTorch, TensorFlow, or Keras | Pretrained weights from ImageNet or any other source |
| Standard layers for convolution, pooling, normalization, dropout, and activation | Fine-tuning or transfer learning from an existing model |
| A hand-written architecture, including a design inspired by VGG or ResNet | Ready-made model constructors such as `torchvision.models.resnet18()` or `keras.applications`, even with random weights |
| Data augmentation, regularization, and learning-rate schedules | Training on outside data or overlapping the training and final evaluation images |
| GPU environments such as Kaggle Notebooks or Google Colab | |

## 6. Notebook evidence

1. **Data loading and preparation.** Show dataset loading, RGB conversion, the fit and internal validation split, normalization, and training augmentation.
2. **Model definition.** Put the architecture in the notebook and show a model summary with the parameter count.
3. **Training.** Show the training loop and report training loss, training accuracy, internal validation loss, and internal validation accuracy for every epoch.
4. **Learning curves.** Plot training and internal validation loss over epochs, and plot both accuracies over epochs.
5. **Final evaluation.** Use the selected checkpoint to predict all 10,000 official evaluation images. Pair each prediction with its matching true label and end the notebook with the evaluation cell below.

Run the notebook from top to bottom and save all cell outputs, so the reported metrics come from the visible training run.

## 7. Final evaluation cell

```python
from sklearn.metrics import accuracy_score, f1_score

# y_true: true labels of the 10,000 evaluation images
# y_pred: the model's predicted labels for the same images
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")

print(f"Evaluation accuracy: {acc:.4f}")
print(f"Evaluation macro F1: {f1:.4f}")
```

| Metric on the official evaluation split | What it measures |
| --- | --- |
| Accuracy | Fraction of images assigned the correct class. |
| Macro F1 | Unweighted mean of the F1 scores of all 200 classes. Each class contributes equally. |

Report both scores to four decimal places, from the notebook's visible training run. [Research notebook](../notebooks/tiny-imagenet-classifier.ipynb)

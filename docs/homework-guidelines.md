# Homework 2 guidelines

Inha University in Tashkent · Artificial Intelligence [202602-SOC4040] · Fall 2026

This guide preserves the requirements from version 3 of the course handout. It applies to the classroom submission. See the [experiment notes](../EXPLANATION.md) for a walkthrough and the [project requirements](requirements.md) for the system built around the homework.

| Item | Requirement |
| --- | --- |
| Task | Predict one of 200 Tiny ImageNet classes for each 64 × 64 image. |
| Model | Design and train a deep learning model, primarily a CNN, from scratch. |
| Grading | Accuracy is worth 50 points and macro F1 is worth 50 points. |
| Deliverables | Submit the executed Jupyter notebook and its PDF export to Homework-2 on eClass. |
| Deadline | 12:00 a.m. on Monday, 30 November 2026. Sunday, 29 November is the last day to submit. |
| Announcements | Follow the course Discord server. |

## 1. Model objective

Write the architecture yourself and train all learned weights from random initialization using Tiny ImageNet's official training data. The model predicts exactly one class label for each input. The official evaluation split supplies the two metrics used for grading.

## 2. Dataset and split roles

Tiny ImageNet is a subset of ImageNet prepared for Stanford CS231n. It has 200 classes and 64 × 64 images. Convert every image to RGB before preprocessing because the set includes some grayscale images.

| Split | Size | Role |
| --- | ---: | --- |
| Official `train` | 100,000 images, 500 per class | Source for both the fit and internal validation portions. Keep the two portions disjoint. |
| Official `valid` (Stanford `val/`) | 10,000 images, 50 per class | Final evaluation only. Use it once, after choosing the architecture, settings, and checkpoint. |
| Official test | 10,000 images | Excluded from the assignment. The labels are private. |

A **90/10 split of official `train`** is the handout's example. A stratified split at that ratio gives you 90,000 fit images and 10,000 internal validation images, with 450 and 50 images per class respectively. Use internal validation to tune the model and choose when to stop training. Save the split seed or indices if you want to reproduce the split.

## 3. Download options

### Hugging Face (recommended by the handout)

Install `datasets` in your notebook environment, then load the training split first:

```python
from datasets import load_dataset

train_data = load_dataset("zh-plus/tiny-imagenet", split="train")
```

Load `split="valid"` into its own variable in the final evaluation section. For Kaggle Notebooks, the handout says to enable Internet in notebook settings before downloading and use a phone-verified Kaggle account. [Hugging Face dataset](https://huggingface.co/datasets/zh-plus/tiny-imagenet)

### Original Stanford archive

The archive is about 240 MB. After extraction, `tiny-imagenet-200/train/` contains one folder per class. The `tiny-imagenet-200/val/` folder contains the 10,000 evaluation images and `val_annotations.txt` with their labels. Use the same class-to-index mapping for training and evaluation. [Stanford CS231n archive](http://cs231n.stanford.edu/tiny-imagenet-200.zip)

## 4. Data leakage and preprocessing

**Data leakage or cheating in the split earns zero for the homework.** Split the official training data first. Fit all learned preprocessing, including normalization means and standard deviations, on the fit portion only. Apply the same normalization to internal validation and final evaluation. Apply training augmentations only to the fit portion.

The following all violate the official evaluation holdout:

- Moving or copying evaluation images into a development split.
- Training on flipped, cropped, or otherwise augmented copies of evaluation images.
- Merging official training and evaluation images and splitting them again.
- Picking a model, learning rate, epoch, or other setting based on the official evaluation score.

Keep the evaluation dataset separate from the training code. Compute final predictions after selecting the model and finishing the training run.

## 5. Allowed and prohibited methods

| Allowed | Prohibited |
| --- | --- |
| A framework such as PyTorch, TensorFlow, or Keras | Pretrained weights from ImageNet or any other source |
| Standard layers for convolution, pooling, normalization, dropout, and activation | Fine-tuning or transfer learning from an existing model |
| An architecture you implement yourself, including a design inspired by VGG or ResNet | Ready-made model constructors such as `torchvision.models.resnet18()` or `keras.applications`, even with random weights |
| Data augmentation, regularization, and learning-rate schedules | Training on outside data or overlapping the training and final evaluation images |
| Free GPU environments such as Kaggle Notebooks or Google Colab | |

The handout says submissions using pretrained weights, fine-tuning, or ready-made model constructors will be rejected.

## 6. Required notebook evidence

1. **Data loading and preparation.** Show dataset loading, RGB conversion, the fit and internal validation split, normalization, and training augmentation.
2. **Model definition.** Put the architecture in the notebook and show a model summary with the parameter count.
3. **Training.** Show the training loop and report training loss, training accuracy, internal validation loss, and internal validation accuracy for every epoch.
4. **Learning curves.** Plot training and internal validation loss over epochs, and plot both accuracies over epochs.
5. **Final evaluation.** Use the selected checkpoint to predict all 10,000 official evaluation images. Pair each prediction with its matching true label and end the notebook with the grading cell below.

Run the notebook from top to bottom and save all cell outputs. The grader uses metrics produced by training visible in the notebook.

## 7. Required final evaluation cell

```python
from sklearn.metrics import accuracy_score, f1_score

# y_true: true labels of the 10,000 evaluation images
# y_pred: your model's predicted labels for the same images
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")

print(f"Evaluation accuracy: {acc:.4f}")
print(f"Evaluation macro F1: {f1:.4f}")
```

## 8. Grading

| Metric on the official evaluation split | Points | What it measures |
| --- | ---: | --- |
| Accuracy | 50 | Fraction of images assigned the correct class. |
| Macro F1 | 50 | Unweighted mean of the F1 scores of all 200 classes. Each class contributes equally. |
| Total | 100 | |

Both scores must be visible to four decimal places. Report scores generated by the notebook's visible training run.

## 9. Submission checklist

The submission area on eClass opens **24 November 2026**. Submit to **Homework-2** by the end of **Sunday, 29 November 2026**.

- [ ] Name the executed notebook `StudentID_HW2.ipynb`, replacing `StudentID` with your own ID.
- [ ] Export that same notebook to `StudentID_HW2.pdf` with the same visible outputs.
- [ ] Open both files and check that the model summary, epoch logs, curves, and final scores are present and match.
- [ ] Upload both files to eClass.

Incorrect filenames incur a **20% deduction from the score earned**. For example, an earned score of 80 becomes 64.

The repository notebook uses a descriptive filename. Apply the required student-ID naming to your submission copies. [Repository notebook](../notebooks/tiny-imagenet-classifier.ipynb)

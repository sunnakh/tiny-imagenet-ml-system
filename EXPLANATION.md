# Tiny ImageNet classification · project overview

## Task

Design a convolutional neural network (CNN), train it from random initialization on Tiny ImageNet, and measure how well it assigns one of 200 class labels to each unseen 64 × 64 image.

## Approach

**Start with the split.** The dataset calls one split `valid`, but it is reserved for the final evaluation. A separate validation set is needed to make decisions while developing the model. A 90/10 split stratified by class gives the following counts.

| Source | Images | How they are used |
| --- | ---: | --- |
| Training portion of `train` | 90,000 · 450 per class | Estimate normalization statistics, augment images and update weights. |
| Validation portion of `train` | 10,000 · 50 per class | Compare architectures, choose an epoch and tune settings. |
| Official `valid` (Stanford `val/`) | 10,000 · 50 per class | Evaluate the chosen model once, after all decisions are final. |
| Official test set | 10,000 · labels unavailable | Not used. |

The project uses the Hugging Face dataset `zh-plus/tiny-imagenet`. The Stanford archive also works. With the archive, use `val_annotations.txt` for final labels and keep the class-to-index mapping consistent with training. [Training protocol](docs/training-protocol.md)

**The notebook follows this order.**

1. **Prepare the inputs.** Split the 100,000 original training images before augmentation. Record the split seed or indices so the split can be reproduced. Convert every image to RGB, including grayscale images. Compute channel means and standard deviations on the training portion and reuse them for validation and final evaluation. Apply random augmentation only during training. If a crop and its original land in different splits, the validation score can reward recognition of the same image.
2. **Define the CNN.** Convolution filters learn patterns shared across image locations. The classifier turns the resulting features into 200 class scores per image. The layers are assembled in the notebook, which displays the architecture and parameter count. Standard layers, regularization, learning-rate schedules and ideas from known architectures are used. Pretrained weights, fine-tuning, external training data and ready-made model constructors are excluded, including constructors called with random weights.
3. **Train and choose a model.** Training batches update the weights. At each epoch, the model is evaluated on the validation portion with updates disabled and the model in evaluation mode. Loss and accuracy are recorded for both portions. For example, if training loss keeps falling while validation loss rises, the earlier epoch is a candidate to keep. Only the training and internal validation results drive that choice.
4. **Plot the learning curves.** Training and validation loss against epoch on one plot, and training and validation accuracy against epoch on another. The plots come from the same run as the evaluated model.
5. **Run the final evaluation once.** Choices are frozen before using official `valid`. Each image gets one predicted class and its matching true label, and the notebook ends with the evaluation cell from the [training protocol](docs/training-protocol.md#7-final-evaluation-cell). Accuracy measures the fraction of correct predictions. Macro F1 computes F1 for each class and averages the 200 scores equally, so it accounts for both missed examples and incorrect predictions of each class. Using the final score to pick an epoch or try another model would leak the evaluation set.

**Local setup.** From this folder, activate the Python environment with `source .venv/bin/activate`, install the dependencies with `uv pip install -r requirements.txt`, then run `jupyter lab`. Kaggle and Colab also work for GPU compute.

**Experiment note (25 November 2026).** The preferred setting is `EPOCHS = 75`. The saved output comes from a run configured for 80 epochs. At epoch 75, training accuracy was 99.39% and internal tuning accuracy was 51.11%. The highest tuning accuracy in the saved log was 51.23% at epoch 77. A run configured for 75 epochs uses a different cosine learning-rate schedule, so its result is still unverified. [Training log](notebooks/tiny-imagenet-classifier.ipynb)

## Done when

- [ ] The two development subsets have disjoint original image IDs and together account for all 100,000 training images. Preprocessing and model-selection steps use only those subsets.
- [ ] A grayscale training image produces three 64 × 64 channels. A batch of size `B` produces `B × 200` model scores, with one class label per image.
- [ ] The saved notebook runs top to bottom and shows the parameter count, all four per-epoch measurements, and both curves. The official evaluation step is kept for the final run.
- [ ] The final output uses 10,000 aligned label–prediction pairs and displays both evaluation accuracy and macro F1 to four decimal places. The training outputs that produced those scores are visible.

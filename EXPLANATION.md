# Homework 2 · Tiny ImageNet classification

## Task

Design your own convolutional neural network (CNN), train it from random initialization on Tiny ImageNet, and measure how well it assigns one of 200 class labels to each unseen 64 × 64 image.

## Requirements

**Start with the split.** The dataset calls one split `valid`, but the instructor reserves it for final grading. You need a separate validation set to make decisions while developing your model. For example, a 90/10 split stratified by class gives the following counts.

| Source | Images | How you may use them |
| --- | ---: | --- |
| Your training portion of `train` | 90,000 · 450 per class | Estimate normalization statistics, augment images and update weights. |
| Your validation portion of `train` | 10,000 · 50 per class | Compare architectures, choose an epoch and tune settings. |
| Official `valid` (Stanford `val/`) | 10,000 · 50 per class | Evaluate the chosen model once, after all decisions are final. |
| Official test set | 10,000 · labels unavailable | Skip it. |

The course recommends the Hugging Face dataset `zh-plus/tiny-imagenet`. The Stanford archive is also allowed. If you choose the archive, use `val_annotations.txt` for final labels and keep the class-to-index mapping consistent with training. [Homework 2 guidelines](docs/homework-guidelines.md)

**Build your notebook in this order.**

1. **Prepare the inputs.** Split the 100,000 original training images before augmentation. Record the split seed or indices so you can reproduce it. Convert every image to RGB, including grayscale images. Compute channel means and standard deviations on your training portion and reuse them for validation and final evaluation. Apply random augmentation only during training. If a crop and its original land in different splits, the validation score can reward recognition of the same image.
2. **Define the CNN yourself.** Convolution filters learn patterns shared across image locations. Your classifier turns the resulting features into 200 class scores per image. Choose and assemble the layers in the notebook, then display the architecture and parameter count. Standard layers, regularization, learning-rate schedules and ideas from known architectures are allowed. Pretrained weights, fine-tuning, external training data and ready-made model constructors are prohibited, including constructors called with random weights.
3. **Train and choose a model.** Use training batches for weight updates. At each epoch, evaluate on your own validation portion with updates disabled and the model in evaluation mode. Record loss and accuracy for both portions. For example, if training loss keeps falling while validation loss rises, the earlier epoch is a candidate to keep. Use only the training and internal validation results to make that choice.
4. **Plot the learning curves.** Show training and validation loss against epoch on one plot, and training and validation accuracy against epoch on another. These plots should come from the same run as the model you evaluate.
5. **Run the final evaluation once.** Freeze your choices before using official `valid`. Collect one predicted class and its matching true label for each image, then finish with the exact evaluation cell supplied in §7 of the PDF. Accuracy measures the fraction of correct predictions. Macro F1 computes F1 for each class and averages the 200 scores equally, so it accounts for both missed examples and incorrect predictions of each class. Each metric is worth 50 points. Evaluation leakage earns zero, including using the final score to pick an epoch or try another model.

**Local setup.** From this folder, activate the Python environment with `source .venv/bin/activate`, install the dependencies with `uv pip install -r requirements.txt`, then run `jupyter lab`. Kaggle and Colab are also allowed if you want GPU compute.

**Experiment note (25 November 2026).** The preferred setting is `EPOCHS = 75`. The saved output comes from a run configured for 80 epochs. At epoch 75, training accuracy was 99.39% and internal tuning accuracy was 51.11%. The highest tuning accuracy in the saved log was 51.23% at epoch 77. A run configured for 75 epochs uses a different cosine learning-rate schedule, so its result is still unverified. [Training log](notebooks/tiny-imagenet-classifier.ipynb)

## Done when

- [ ] Your two development subsets have disjoint original image IDs and together account for all 100,000 training images. Your preprocessing and model-selection steps use only those subsets.
- [ ] A grayscale training image produces three 64 × 64 channels. A batch of size `B` produces `B × 200` model scores, with one class label per image.
- [ ] The saved notebook runs top to bottom and shows the parameter count, all four per-epoch measurements, and both curves. Keep the official evaluation step for the final run.
- [ ] The final output uses 10,000 aligned label–prediction pairs and displays both evaluation accuracy and macro F1 to four decimal places. The training outputs that produced those scores are visible.
- [ ] Export the executed notebook to PDF and open it to check that the curves and final scores are visible and match the notebook. Upload **`StudentID_HW2.ipynb`** and **`StudentID_HW2.pdf`** to Homework-2 on eClass, replacing `StudentID` with your own ID. Incorrect naming costs 20% of the score earned.
- [ ] Submit by the end of **Sunday, 29 November 2026**. The stated cutoff is midnight at the start of Monday, 30 November.

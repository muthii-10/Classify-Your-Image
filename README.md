# Classify-Your-Image

A Python-based image classification project that trains a deep learning model to recognize different image classes and provides a command-line interface for inference using trained checkpoints. This repository leverages PyTorch for building, training, and deploying image classifiers.

---

## Overview

This project implements a workflow that allows you to:

1. Preprocess image datasets
2. Train a convolutional neural network (CNN) using a pre-trained model backbone
3. Save model checkpoints
4. Run predictions directly from the command line

This setup makes it easy to train custom classifiers on your own image datasets with minimal configuration.

---

## 📦 Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

---

### 🛠️ Features

- Train an image classification model with **transfer learning** (e.g., `vgg16`, `alexnet`)  
- Load and preprocess image datasets with augmentations  
- Save trained checkpoints  
- Predict top‑K classes for a given image  
- Supports category‑to‑name mappings for human‑readable labels  

---

## 📌 Training the Model

Prepare a dataset folder (e.g., flowers/) with subfolders train/, valid/, and test/, each containing subfolders for each class. Then run:

```bash
python train.py \
  --data_dir flowers \
  --save_dir checkpoints \
  --arch vgg16 \
  --learning_rate 0.001 \
  --hidden_units 512 \
  --epochs 5 \
  --gpu
  ```
  This will train the model and save a checkpoint to the given directory.

  ## 📌 Running Predictions

To classify a new image, use:
```bash
python predict.py \
  --input 'path_to_image.jpg' \
  --checkpoint 'checkpoints/checkpoint.pth' \
  --top_k 5 \
  --category_names 'cat_to_name.json' \
  --gpu
  ```
  ## 📜 License

This project is open-source.

## 👤 Author

Victor Muthii

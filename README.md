# Text Classification with TensorFlow Hub Embeddings

This project is a **binary text classification model** trained on the IMDb movie reviews dataset. It uses **TensorFlow Hub's pre-trained embedding layer** to convert input text into numerical vectors, followed by a dense neural network for sentiment classification (positive or negative review).

## 📂 Dataset

- **Source**: IMDb Reviews (from TensorFlow Datasets)
- **Splits**:
  - Training: 60% of `train`
  - Validation: 40% of `train`
  - Testing: `test` split from dataset

## 🧠 Model Architecture

- **Embedding Layer**: TensorFlow Hub's `nnlm-en-dim50` embedding model
- **Hidden Layer**: `Dense(16, activation='relu')`
- **Output Layer**: `Dense(1, activation='sigmoid')` for binary classification

## ⚙️ Training Details

- **Loss**: `BinaryCrossentropy(from_logits=True)`
- **Optimizer**: `Adam`
- **Metrics**: `accuracy`
- **Batch Size**: 100
- **Epochs**: 25

## 🔍 Evaluation

The model was evaluated using accuracy and binary cross-entropy loss. Predictions were converted to class labels using a sigmoid threshold (0.5).


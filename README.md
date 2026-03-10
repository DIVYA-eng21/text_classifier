# IMDB Text Classification

This project performs **sentiment classification on movie reviews** using deep learning models built with TensorFlow.  
The objective is to classify reviews as **positive** or **negative**.

The project originally used **TensorFlow Hub embeddings**, and it has now been expanded to include **Recurrent Neural Network (RNN) based models** for sequential text learning.

---

# Dataset

The dataset used is the **IMDB Reviews Dataset**, loaded using TensorFlow Datasets.


The dataset consists of **movie reviews as text** and **binary sentiment labels**:

- `0` → Negative review  
- `1` → Positive review

---

# Implemented Models

This repository now includes **multiple deep learning approaches for text classification**.

## 1. TensorFlow Hub Embedding Model

This model uses **pretrained embeddings from TensorFlow Hub** to convert text into dense vector representations before classification.

Pipeline:

```
Text → Pretrained Embedding (TF Hub) → Dense Layers → Output
```

Advantages:
- Uses pretrained semantic knowledge
- Faster convergence
- Good baseline model for text classification

---

## 2. RNN with Integer Encoding

In this approach:

1. Text is **tokenized**
2. Each word is converted into an **integer index**
3. Sequences are **padded to equal length**
4. The sequences are passed directly to an **RNN**

Pipeline:

```
Text → Tokenization → Integer Encoding → RNN → Dense → Output
```

RNNs process text **sequentially**, allowing the model to capture **word order and contextual dependencies**.

---

## 3. Embedding + RNN Model

This model combines **trainable embeddings with a recurrent neural network**.

Pipeline:

```
Text → Integer Encoding → Embedding Layer → RNN → Dense → Output
```

Explanation:

- **Embedding Layer** learns vector representations of words during training.
- **RNN** captures sequential patterns and contextual relationships between words.

This approach allows the model to learn both:
- semantic representations of words
- sequential language structure

---

# Project Structure

```
text_classifier/
│
├── text.ipynb
├── Sentiment_an_integerencoding(RNN).ipynb
├── Sentiment_an_embedding(RNN).ipynb
├── README.md
```

Description of files:

- **text.ipynb**  
  Text classification using TensorFlow Hub pretrained embeddings.

- **Sentiment_an_embedding(RNN).ipynb**  
  RNN model trained using integer encoded text sequences.

- **Sentiment_an_integerencoding(RNN).ipynb**  
  RNN model that uses a trainable embedding layer before the recurrent network.

---

# How to Run the Project

If you want to run this project locally, follow these steps.

### 1. Clone the repository

```
git clone https://github.com/DIVYA-eng21/text_classifier.git
```

### 2. Move into the project directory

```
cd text_classifier
```

### 3. Install required dependencies

```
pip install -r requirements.txt
```

### 4. Open the notebooks

Run the notebooks using **Jupyter Notebook** or **Google Colab**.

---

# Technologies Used

- Python
- TensorFlow
- TensorFlow Datasets
- NumPy
- Deep Learning (RNN)

---

# Future Improvements

Possible improvements include:

- Using **LSTM or GRU networks**
- Using **transformer-based models**
- Hyperparameter tuning
- Adding attention mechanisms



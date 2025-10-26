# Semantic Textual Similarity (STS) Modeling

This project focuses on predicting the **semantic similarity** between pairs of sentences —  
a key NLP task used in semantic search, question answering, and duplicate detection.

It explores multiple approaches, from classical ML to deep learning and transformer-based models.

---

## 🎯 Objective

Given two sentences, the model learns to output a similarity score representing how semantically related they are.  
This task is also known as **Semantic Textual Similarity (STS)** or **Sentence Matching**.

---

## 📁 Dataset

| File | Description |
|------|--------------|
| `train_prompts.csv` | Training prompts |
| `train_responses.csv` | Corresponding responses and similarity labels |
| `dev_prompts.csv` | Validation prompts |
| `dev_responses.csv` | Validation responses and labels |

Each row contains a **sentence pair** and a **semantic similarity score** (or binary label).

---

## 🧠 Methodology

- **Track 1:** Classical NLP  
  TF-IDF vectorization + Cosine similarity or Logistic Regression baseline

- **Track 2:** Deep Learning  
  Siamese / BiLSTM network using pre-trained word embeddings (GloVe / Word2Vec)

- **Track 3:** Transformer-based  
  Fine-tuning BERT / Sentence-BERT for sentence pair regression or classification

---

## 📊 Evaluation

Depending on the formulation (regression or classification):

- **Regression:** Pearson correlation, Mean Squared Error (MSE)  
- **Classification:** Accuracy, F1-score  

---

## ⚙️ Tech Stack

Python, Pandas, NumPy, scikit-learn, PyTorch, HuggingFace Transformers, NLTK




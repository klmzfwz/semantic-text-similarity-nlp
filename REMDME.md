# Semantic Textual Similarity Modeling

This project aims to predict the semantic similarity between pairs of sentences using NLP models.  
The dataset contains paired prompts and responses with similarity labels.

## Dataset
| File | Description |
|------|--------------|
| train_prompts.csv | Training set prompts |
| train_responses.csv | Corresponding responses and labels |
| dev_prompts.csv | Validation set prompts |
| dev_responses.csv | Validation set responses and labels |

## Approach
- **Track 1:** Classical methods (TF-IDF + cosine similarity)
- **Track 2:** Siamese LSTM and MLP architectures
- **Track 3:** Transformer-based models (BERT, SBERT)

## Metrics
- Pearson correlation (for regression)
- F1 / Accuracy (for classification)

## Tech Stack
Python, Pandas, scikit-learn, PyTorch, HuggingFace Transformers

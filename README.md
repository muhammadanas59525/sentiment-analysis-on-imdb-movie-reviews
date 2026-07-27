# Sentiment Analysis on IMDB Movie Reviews

## Project Overview

This project builds a **sentiment analysis model** that classifies IMDB movie reviews as either **positive** or **negative** using Natural Language Processing (NLP) techniques.

## Dataset

- **Source:** IMDB Movie Reviews Dataset (999 reviews)
- **Labels:** `positive` and `negative` (501 positive, 498 negative — nearly balanced)
- **Format:** CSV with `review` (text) and `sentiment` (label) columns

## Pipeline Steps

### 1. Data Loading & Inspection
- Loaded the dataset using pandas
- Checked class balance (50.2% positive, 49.8% negative)

### 2. Text Preprocessing
- Converted all text to lowercase
- Removed HTML tags (e.g., `<br />`)
- Removed punctuation and special characters using regex
- Removed extra whitespace

### 3. Feature Extraction (TF-IDF)
- Used `TfidfVectorizer` from scikit-learn
- Parameters: `max_features=5000`, `ngram_range=(1, 2)` (unigrams + bigrams)
- English stop words removed automatically

### 4. Model Training
- **Logistic Regression:** A linear model well-suited for high-dimensional sparse data
- **Multinomial Naive Bayes:** A probabilistic classifier effective for text data

### 5. Evaluation
- **Accuracy:** Percentage of correct predictions
- **F1-Score:** Harmonic mean of precision and recall

## Results

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | ~79% | ~0.80 |

## Custom Test Predictions

| Sentence | Predicted Sentiment |
|----------|-------------------|
| "This was the best experience ever! I absolutely loved every moment of it." | POSITIVE |
| "This was a complete waste of time. Terrible acting and boring plot." | NEGATIVE |
| "The movie was okay, not great but not terrible either. Some parts were good." | POSITIVE |

## Limitations

1. **Sarcasm Detection:** The model struggles with sarcastic or ironic statements
2. **Context Understanding:** TF-IDF treats words independently and cannot capture negation well (e.g., "not bad")
3. **Vocabulary Limitation:** With `max_features=5000`, rare but important words may be excluded
4. **Domain Specificity:** Trained on movie reviews; may not generalize to other domains
5. **No Word Order:** Bag-of-words approach loses sequential information
6. **Small Dataset Size:** The dataset contains 999 reviews, which limits the model's ability to generalize

## How to Run

### Prerequisites
- Python 3.8+
- Required libraries: `scikit-learn`, `nltk`, `pandas`, `jupyter`

### Installation
```bash
pip install scikit-learn nltk pandas jupyter
```

### Running the Notebook
```bash
jupyter notebook sentiment_analysis.ipynb
```

Run all cells sequentially from top to bottom.

### Running the Test Script
```bash
python test_script.py
```

## Files

- `sentiment_analysis.ipynb` - Complete Jupyter notebook with the full pipeline
- `test_script.py` - Standalone Python script that runs the full pipeline
- `IMDB Dataset.csv` - The dataset (999 IMDB movie reviews)
- `README.md` - This file
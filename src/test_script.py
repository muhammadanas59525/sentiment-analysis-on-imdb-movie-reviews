import pandas as pd
import numpy as np
import re
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'IMDB Dataset.csv')

df = pd.read_csv(csv_path)
print(f'Dataset shape: {df.shape}')
print('Sentiment distribution:')
print(df['sentiment'].value_counts())

# Quick test of cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_review'] = df['review'].apply(clean_text)
sample = df['cleaned_review'].iloc[0][:100]
print(f'\nCleaning applied. Sample: {sample}')

# Quick TF-IDF test
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X = vectorizer.fit_transform(df['cleaned_review'])
df['sentiment_label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
y = df['sentiment_label'].values
print(f'TF-IDF matrix shape: {X.shape}')

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print('\nLOGISTIC REGRESSION RESULTS')
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'F1-Score: {f1_score(y_test, y_pred):.4f}')

# Test custom sentences
test_sentences = [
    'This was the best experience ever I absolutely loved every moment of it',
    'This was a complete waste of time terrible acting and boring plot',
    'The movie was okay not great but not terrible either some parts were good'
]
test_vec = vectorizer.transform(test_sentences)
preds = lr.predict(test_vec)
for i, s in enumerate(test_sentences):
    sent = 'POSITIVE' if preds[i] == 1 else 'NEGATIVE'
    print(f'Sentence: {s[:60]}... -> {sent}')

print('\nAll tests passed successfully!')
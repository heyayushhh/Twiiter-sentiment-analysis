# Twitter Sentiment Analysis using Logistic Regression

This project performs sentiment analysis on a Twitter dataset (Sentiment140) using natural language processing and machine learning techniques in Python (Google Colab). The dataset contains 1.6 million tweets classified into positive and negative sentiments.

## 📁 Dataset

- **Name**: Sentiment140
- **Source**: [Kaggle - Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Size**: 1.6 million tweets
- **Format**: CSV file with 6 columns:
  - `target`: Sentiment label (0 = Negative, 4 = Positive)
  - `id`: Tweet ID
  - `date`: Date of tweet
  - `flag`: Query flag
  - `user`: User handle
  - `text`: Actual tweet

## 🚀 Project Workflow

### 1. Setup & Installation
- Install `kaggle` package
- Configure Kaggle API token to download dataset

### 2. Data Loading
- Load the CSV file into a Pandas DataFrame
- Rename columns appropriately

### 3. Preprocessing
- Replace target label `4` with `1` for binary classification (0 = Negative, 1 = Positive)
- Clean and normalize tweet text:
  - Remove non-alphabetic characters
  - Lowercase conversion
  - Tokenization and stopword removal
  - Stemming using NLTK’s `PorterStemmer`

### 4. Feature Extraction
- Use `TfidfVectorizer` to convert cleaned text into numerical feature vectors

### 5. Model Building
- Split the dataset into training and testing sets (80-20 split)
- Train a Logistic Regression model
- Evaluate the model:
  - Training Accuracy: **~79.87%**
  - Testing Accuracy: **~77.67%**

### 6. Model Saving
- Save the trained model using `pickle` for future use

## 🛠️ Technologies Used

- Python (Google Colab)
- Pandas, NumPy
- NLTK (Natural Language Toolkit)
- Scikit-learn
- TfidfVectorizer
- Logistic Regression
- Pickle

## ✅ Results

The Logistic Regression model performs decently well with nearly 78% accuracy on unseen test data. It demonstrates how simple models can still provide valuable insights on large-scale textual data.

## 🔖 License

Dataset License: Other (as provided by Kaggle)

---

**Author**: *Ayush Singh*


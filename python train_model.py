"""
Script to train and save the sentiment analysis model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pickle

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words
        tokens = [t for t in tokens if t not in self.stop_words]
        
        return ' '.join(tokens)

class SentimentAnalyzer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(max_iter=1000)
    
    def prepare_data(self, df):
        """Prepare data for training"""
        print("Initial data shape:", df.shape)
        
        # Clean data
        df = df.copy()
        df = df.dropna(subset=['Tweets', 'Sentiment'])
        
        print("Cleaning texts...")
        df['cleaned_tweets'] = df['Tweets'].apply(self.preprocessor.clean_text)
        df = df[df['cleaned_tweets'].str.len() > 0]
        
        # Convert sentiment to numeric
        sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        df['sentiment_numeric'] = df['Sentiment'].map(sentiment_map)
        df = df.dropna(subset=['sentiment_numeric'])
        
        print("Final shape:", df.shape)
        print("\nSentiment distribution:")
        print(df['sentiment_numeric'].value_counts())
        
        return train_test_split(
            df['cleaned_tweets'].values,
            df['sentiment_numeric'].values,
            test_size=0.2,
            random_state=42,
            stratify=df['sentiment_numeric'].values
        )
    
    def train(self, X_train, y_train):
        """Train the model"""
        print("\nTraining model...")
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vectorized, y_train)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        X_test_vectorized = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_vectorized)
        return classification_report(y_test, y_pred,
                                  target_names=['Negative', 'Neutral', 'Positive'])

def main():
    # Load your CSV file
    print("Loading data...")
    df = pd.read_csv('Final Tweets.csv')  # Make sure your CSV file is in the same directory
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Prepare and train
    X_train, X_test, y_train, y_test = analyzer.prepare_data(df)
    analyzer.train(X_train, y_train)
    
    # Evaluate
    report = analyzer.evaluate(X_test, y_test)
    print("\nClassification Report:")
    print(report)
    
    # Save the trained model
    print("\nSaving model...")
    with open('sentiment_model.pkl', 'wb') as f:
        pickle.dump(analyzer, f)
    print("Model saved successfully!")

if __name__ == "__main__":
    main()

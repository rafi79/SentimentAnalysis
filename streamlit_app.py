"""
Streamlit Sentiment Analysis App
"""

import streamlit as st
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
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')

download_nltk_data()

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
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtag symbols but keep text
        text = re.sub(r'#', '', text)
        
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
        # Clean texts
        df = df.copy()
        df = df.dropna(subset=['Tweets', 'Sentiment'])
        
        df['cleaned_tweets'] = df['Tweets'].apply(self.preprocessor.clean_text)
        df = df[df['cleaned_tweets'].str.len() > 0]
        
        # Convert sentiment to numeric
        sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        df['sentiment_numeric'] = df['Sentiment'].map(sentiment_map)
        df = df.dropna(subset=['sentiment_numeric'])
        
        return train_test_split(
            df['cleaned_tweets'].values,
            df['sentiment_numeric'].values,
            test_size=0.2,
            random_state=42,
            stratify=df['sentiment_numeric'].values
        )
    
    def train(self, X_train, y_train):
        """Train the model"""
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vectorized, y_train)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        X_test_vectorized = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_vectorized)
        
        return classification_report(y_test, y_pred,
                                  target_names=['Negative', 'Neutral', 'Positive'])
    
    def predict(self, text):
        """Predict sentiment for a single text"""
        cleaned_text = self.preprocessor.clean_text(text)
        X = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        return {
            'sentiment': sentiment_map[prediction],
            'confidence': float(max(probabilities)),
            'cleaned_text': cleaned_text
        }

def main():
    st.title("Sentiment Analysis App")
    st.write("Upload a CSV file to train the model or use the pre-trained model for predictions")
    
    # Initialize session state for model
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    
    # File upload section
    st.header("Train Model")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.write(df.head())
            
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    analyzer = SentimentAnalyzer()
                    X_train, X_test, y_train, y_test = analyzer.prepare_data(df)
                    analyzer.train(X_train, y_train)
                    report = analyzer.evaluate(X_test, y_test)
                    
                    st.session_state.analyzer = analyzer
                    st.success("Model trained successfully!")
                    
                    st.subheader("Model Performance")
                    st.text(report)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Text input section
    st.header("Predict Sentiment")
    text_input = st.text_area("Enter text for sentiment analysis:")
    
    if st.button("Analyze Sentiment"):
        if st.session_state.analyzer is None:
            st.warning("Please train the model first!")
        else:
            try:
                result = st.session_state.analyzer.predict(text_input)
                
                # Display results with color-coding
                sentiment_color = {
                    'positive': 'green',
                    'neutral': 'blue',
                    'negative': 'red'
                }
                
                st.markdown(f"**Sentiment:** :{sentiment_color[result['sentiment']]}[{result['sentiment'].upper()}]")
                st.markdown(f"**Confidence:** {result['confidence']:.2%}")
                
                # Show cleaned text
                with st.expander("See preprocessed text"):
                    st.write(result['cleaned_text'])
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()

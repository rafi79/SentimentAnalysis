"""
Complete Streamlit Sentiment Analysis App with Training and Prediction
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
        # Clean data
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
        
        if not cleaned_text:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'cleaned_text': cleaned_text
            }
        
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
    st.title("AI Tweet Sentiment Analysis")
    
    # Initialize session state for model
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
        st.session_state.model_trained = False
    
    # File upload section
    st.header("Train Model")
    uploaded_file = st.file_uploader("Upload CSV file (must contain 'Tweets' and 'Sentiment' columns)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.write(df.head())
            
            if 'Tweets' not in df.columns or 'Sentiment' not in df.columns:
                st.error("CSV must contain 'Tweets' and 'Sentiment' columns!")
                return
            
            if st.button("Train Model"):
                with st.spinner("Training model... This may take a few minutes."):
                    # Initialize and train model
                    analyzer = SentimentAnalyzer()
                    X_train, X_test, y_train, y_test = analyzer.prepare_data(df)
                    analyzer.train(X_train, y_train)
                    report = analyzer.evaluate(X_test, y_test)
                    
                    # Save in session state
                    st.session_state.analyzer = analyzer
                    st.session_state.model_trained = True
                    
                    st.success("Model trained successfully!")
                    
                    # Show evaluation metrics
                    st.subheader("Model Performance")
                    st.text(report)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Text analysis section
    st.header("Analyze Sentiment")
    text_input = st.text_area("Enter text to analyze:")
    
    if st.button("Analyze Sentiment"):
        if not st.session_state.model_trained:
            st.warning("Please train the model first by uploading a CSV file!")
        elif not text_input:
            st.warning("Please enter some text to analyze!")
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
    
    # Example usage
    with st.expander("See example texts"):
        st.write("""
        Try these example texts:
        - "AI has revolutionized the way we solve complex problems!"
        - "This AI implementation is quite disappointing."
        - "The AI system performs as expected."
        """)

if __name__ == "__main__":
    main()

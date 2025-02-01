"""
Streamlit app for sentiment analysis using pre-trained model
"""
import streamlit as st
import pandas as pd
import pickle

def load_model():
    try:
        with open('sentiment_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

def main():
    st.title("AI Tweet Sentiment Analysis")
    
    # Load the pre-trained model
    analyzer = load_model()
    
    if analyzer is None:
        st.error("No trained model found! Please run the training script first.")
        return
    
    # Text input section
    st.header("Analyze Sentiment")
    text_input = st.text_area("Enter text for sentiment analysis:", height=100)
    
    if st.button("Analyze"):
        if not text_input:
            st.warning("Please enter some text to analyze.")
        else:
            try:
                result = analyzer.predict(text_input)
                
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

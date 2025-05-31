import streamlit as st
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

# App Header
st.title("🧠 Product Review Sentiment Analyzer by Abdul Aziz")
st.markdown("Analyze the sentiment of your product reviews in real-time.")

# --- Text Review Input ---
st.subheader("📝 Analyze a Single Review")
review = st.text_area("Enter your product review here:")

if st.button("🔍 Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review before analyzing.")
    else:
        blob = TextBlob(review)
        sentiment_score = blob.sentiment.polarity

        if sentiment_score > 0:
            st.success(f"😊 Positive Sentiment! (Score: {sentiment_score:.2f})")
        elif sentiment_score < 0:
            st.error(f"😠 Negative Sentiment! (Score: {sentiment_score:.2f})")
        else:
            st.info(f"😐 Neutral Sentiment (Score: {sentiment_score:.2f})")

# --- Upload CSV Section ---
st.markdown("---")
st.subheader("📁 Upload CSV File of Reviews")
uploaded_file = st.file_uploader("Upload a CSV file with a `review` column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "review" not in df.columns:
        st.error("The CSV file must contain a 'review' column.")
    else:
        # Apply sentiment analysis
        df['polarity'] = df['review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df['sentiment'] = df['polarity'].apply(
            lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral")
        )

        # Show results
        st.write("📊 Sample Results")
        st.dataframe(df[['review', 'sentiment', 'polarity']].head(10))

        # Chart sentiment counts
        sentiment_counts = df['sentiment'].value_counts()

        st.markdown("### 📈 Sentiment Distribution")
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'], ax=ax)
        plt.title("Sentiment Breakdown")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        st.pyplot(fig)

        # Option to download cleaned result
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Analyzed Data as CSV",
            data=csv,
            file_name='sentiment_results.csv',
            mime='text/csv',
        )

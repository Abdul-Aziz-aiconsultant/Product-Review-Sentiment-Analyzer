
import pandas as pd

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import word_tokenize, pos_tag
from textblob import TextBlob




# Download (run once, you can comment this after first run)
nltk.download("stopwords")
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('brown')
nltk.download('punkt_tab')

# Step 1: Load the dataset
df = pd.read_csv("reviews.csv")  # Make sure 'reviews.csv' is in your project folder

# Step 2: Define a function to clean text
def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation (!, ., ?, etc.)
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize (split sentence into words)
    tokens = word_tokenize(text)

    # Remove stopwords like 'is', 'the', 'and'
    stop_words = set(stopwords.words("english"))
    filtered = [word for word in tokens if word not in stop_words]

    # Re-join words into a string
    return " ".join(filtered)

# Step 3: Apply cleaning function to the review column
df["cleaned_review"] = df["review"].apply(clean_text)

# Step 4: Show the results
print("\n Cleaned Reviews:\n")
print(df[["review", "cleaned_review"]])

df.to_csv("cleaned_reviews.csv", index=False)


def extract_aspects(text):
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    
    # Extract only nouns (NN, NNS, NNP, NNPS)
    aspects = [word for word, tag in tags if tag.startswith('NN')]
    return aspects


df['aspects'] = df['cleaned_review'].apply(extract_aspects)
print(df[['cleaned_review', 'aspects']])    

df.to_csv("reviews_with_aspects.csv", index=False)

def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply it to cleaned reviews
df['sentiment'] = df['cleaned_review'].apply(get_sentiment)
print(df[['cleaned_review', 'sentiment']])

df.to_csv("final_reviews_with_sentiment.csv", index=False)

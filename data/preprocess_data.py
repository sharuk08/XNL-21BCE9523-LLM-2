import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')

STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in STOPWORDS]
    return " ".join(filtered_tokens)

def preprocess_data():
    df = pd.read_csv("data/raw.csv")
    df['texts'] = df['texts'].apply(preprocess_text)
    df['labels'] = df['labels'].apply(lambda x: "positive" if str(x).lower() in ['positive', '4'] else "negative")
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train.to_csv("data/raw_train.csv", index=False)
    test.to_csv("data/raw_test.csv", index=False)
    print("Preprocessed data saved.")

preprocess_data()
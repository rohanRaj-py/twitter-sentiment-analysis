import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download once
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()

    # Remove stopwords + apply stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words]

    return " ".join(words)
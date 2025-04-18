# Write a Python program to clean a paragraph using tokenization, stopword removal, and stemming.

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

def clean_paragraph(paragraph):
    # Initialize stemmer and stopword list
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Step 1: Tokenization
    tokens = word_tokenize(paragraph)

    # Step 2: Remove punctuation and convert to lowercase
    words = [word.lower() for word in tokens if word.isalpha()]

    # Step 3: Stopword removal
    filtered_words = [word for word in words if word not in stop_words]

    # Step 4: Stemming
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    return stemmed_words

# Example usage
paragraph = """Natural Language Processing (NLP) is a fascinating field of Artificial Intelligence.
It enables computers to understand and generate human language."""
cleaned_words = clean_paragraph(paragraph)

print("Cleaned Words:")
print(cleaned_words)

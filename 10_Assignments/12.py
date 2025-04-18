# Implement TF-IDF vectorization on a set of text documents using sklearn.
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text documents
documents = [
    "Natural Language Processing is a branch of artificial intelligence.",
    "Machine learning and NLP are core parts of AI.",
    "TF-IDF stands for Term Frequency-Inverse Document Frequency.",
    "It is used to convert text into numerical feature vectors.",
    "NLP helps computers understand human language."
]

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the feature names (i.e., the vocabulary)
feature_names = vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to a dense format and print
dense_matrix = tfidf_matrix.toarray()

# Display TF-IDF values
print("TF-IDF Feature Names:")
print(feature_names)
print("\nTF-IDF Matrix:")
for i, doc in enumerate(dense_matrix):
    print(f"Document {i+1}:")
    print(doc)

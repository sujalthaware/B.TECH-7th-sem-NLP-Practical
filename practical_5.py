# NLP Lab: Bag of Words and TF-IDF

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Given text corpus
documents = [
    "I love natural language processing",
    "Natural language processing is interesting",
    "I love machine learning"
]

print("Original Documents:")
for i, doc in enumerate(documents, 1):
    print(f"Document {i}: {doc}")

# --------------------------------------------------
# 1. Bag of Words (BoW)
# --------------------------------------------------

bow_vectorizer = CountVectorizer()

# Convert text into BoW numerical vectors
bow_matrix = bow_vectorizer.fit_transform(documents)

print("\n--- Bag of Words (BoW) ---")

# Display vocabulary
print("\nVocabulary:")
print(bow_vectorizer.get_feature_names_out())

# Display numerical vectors
print("\nBoW Matrix:")
print(bow_matrix.toarray())

# --------------------------------------------------
# 2. TF-IDF
# --------------------------------------------------

tfidf_vectorizer = TfidfVectorizer()

# Convert text into TF-IDF numerical vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("\n--- TF-IDF ---")

# Display vocabulary
print("\nVocabulary:")
print(tfidf_vectorizer.get_feature_names_out())

# Display numerical vectors
print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())

# --------------------------------------------------
# 3. Display TF-IDF values clearly
# --------------------------------------------------

print("\nTF-IDF Values:")

for i, row in enumerate(tfidf_matrix.toarray(), 1):
    print(f"Document {i}:")
    for word, value in zip(
        tfidf_vectorizer.get_feature_names_out(), row
    ):
        if value > 0:
            print(f"  {word}: {value:.4f}")
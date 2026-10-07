# NLP Lab: Word2Vec Word Embeddings

import nltk
from gensim.models import Word2Vec

# Download tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize

# --------------------------------------------------
# Step 1: Define the text corpus
# --------------------------------------------------

corpus = [
    "Natural language processing is a field of artificial intelligence",
    "Natural language processing helps computers understand human language",
    "Machine learning is an important part of artificial intelligence",
    "Deep learning is used in natural language processing",
    "Python is widely used for machine learning and NLP",
    "Word embeddings represent words as numerical vectors",
    "Word2Vec learns meaningful representations of words",
    "Natural language processing uses machine learning techniques"
]

# --------------------------------------------------
# Step 2: Tokenize the corpus
# --------------------------------------------------

tokenized_corpus = [
    word_tokenize(sentence.lower())
    for sentence in corpus
]

print("Tokenized Corpus:")
for sentence in tokenized_corpus:
    print(sentence)

# --------------------------------------------------
# Step 3: Train the Word2Vec model
# --------------------------------------------------

model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1
)

# --------------------------------------------------
# Step 4: Display vector representation of a word
# --------------------------------------------------

word = "language"

print("\nVector representation of:", word)
print(model.wv[word])

# --------------------------------------------------
# Step 5: Find similar words
# --------------------------------------------------

print("\nWords similar to 'language':")

similar_words = model.wv.most_similar("language", topn=5)

for word, similarity in similar_words:
    print(f"{word}: {similarity:.4f}")

# --------------------------------------------------
# Step 6: Calculate similarity between two words
# --------------------------------------------------

word1 = "language"
word2 = "processing"

similarity = model.wv.similarity(word1, word2)

print(f"\nSimilarity between '{word1}' and '{word2}':")
print(f"{similarity:.4f}")
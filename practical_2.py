# NLP Lab: Part-of-Speech (POS) Tagging

import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import Counter

# Given text corpus
text = """
Natural language processing is an important field of artificial intelligence.
It allows computers to understand and process human language.
Students are learning NLP techniques using Python.
"""

print("Original Text:")
print(text)

# Step 1: Tokenization
tokens = word_tokenize(text)

print("\nTokens:")
print(tokens)

# Step 2: POS Tagging
pos_tags = pos_tag(tokens)

print("\nPOS Tagged Words:")
for word, tag in pos_tags:
    print(f"{word:15} -> {tag}")

# Step 3: Analyze word categories
pos_counts = Counter(tag for word, tag in pos_tags)

print("\nWord Category Analysis:")
for tag, count in pos_counts.items():
    print(f"{tag:8} : {count}")

# Step 4: Display words belonging to each category
print("\nWords by POS Category:")

categories = {}

for word, tag in pos_tags:
    categories.setdefault(tag, []).append(word)

for tag, words in categories.items():
    print(f"{tag}: {words}")
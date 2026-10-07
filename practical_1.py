# NLP Lab: Tokenization, Stop-word Removal, Stemming and Lemmatization

import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Input text
text = "The students are studying natural language processing and they are learning different techniques."

print("Original Text:")
print(text)

# 1. Tokenization
tokens = word_tokenize(text)

print("\n1. Tokens:")
print(tokens)

# 2. Stop-word Removal
stop_words = set(stopwords.words('english'))

filtered_tokens = [
    word for word in tokens
    if word.lower() not in stop_words and word.isalpha()
]

print("\n2. After Stop-word Removal:")
print(filtered_tokens)

# 3. Stemming
stemmer = PorterStemmer()

stemmed_words = [stemmer.stem(word) for word in filtered_tokens]

print("\n3. After Stemming:")
print(stemmed_words)

# 4. Lemmatization
lemmatizer = WordNetLemmatizer()

lemmatized_words = [
    lemmatizer.lemmatize(word, pos='v')
    for word in filtered_tokens
]

print("\n4. After Lemmatization:")
print(lemmatized_words)
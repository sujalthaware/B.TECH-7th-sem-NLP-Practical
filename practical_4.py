# NLP Lab: Spam Detection using Machine Learning

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------------------------
# Step 1: Create a sample dataset
# --------------------------------------------------

data = {
    "message": [
        "Congratulations! You won a free lottery ticket",
        "Win a cash prize of 10000 now",
        "Claim your free gift card today",
        "You have won a free vacation",
        "Congratulations, you have won a prize",
        "Free entry in a competition, click now",
        
        "Hey, are we meeting for lunch today?",
        "Please send me the assignment",
        "Can you call me when you reach home?",
        "The meeting is scheduled for tomorrow",
        "Don't forget to bring your project file",
        "Happy birthday! Have a great day",
        "Are you coming to college today?",
        "Please submit the report by evening",
        "Let's meet at the library",
        "Your appointment is confirmed for tomorrow"
    ],

    "label": [
        "spam", "spam", "spam", "spam", "spam", "spam",
        "ham", "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham", "ham"
    ]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# --------------------------------------------------
# Step 2: Separate features and labels
# --------------------------------------------------

X = df["message"]
y = df["label"]

# --------------------------------------------------
# Step 3: Split dataset into training and testing
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Step 4: Convert text into TF-IDF features
# --------------------------------------------------

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --------------------------------------------------
# Step 5: Train Naive Bayes classifier
# --------------------------------------------------

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

# --------------------------------------------------
# Step 6: Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test_tfidf)

# --------------------------------------------------
# Step 7: Evaluate the model
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# Step 8: Test with new messages
# --------------------------------------------------

new_messages = [
    "Congratulations! You have won a free prize",
    "Can you send me the notes?",
    "Claim your cash reward now",
    "Let's meet after class"
]

new_messages_tfidf = vectorizer.transform(new_messages)

predictions = model.predict(new_messages_tfidf)

print("\nNew Message Predictions:")

for message, prediction in zip(new_messages, predictions):
    print(f"{message} -> {prediction}")
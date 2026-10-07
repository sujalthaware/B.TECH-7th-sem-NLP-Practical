# NLP Lab: Sentiment Classification using DistilBERT

# Install required libraries first:
# pip install transformers datasets torch scikit-learn

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# --------------------------------------------------
# Step 1: Load the IMDb review dataset
# --------------------------------------------------

dataset = load_dataset("imdb")

print("Dataset:")
print(dataset)

# Use a smaller subset for a simple lab experiment
train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))
test_dataset = dataset["test"].shuffle(seed=42).select(range(500))


# --------------------------------------------------
# Step 2: Load DistilBERT tokenizer
# --------------------------------------------------

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)


# --------------------------------------------------
# Step 3: Tokenize the reviews
# --------------------------------------------------

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# Remove original text column
train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

# Rename label column if required
train_dataset.set_format("torch")
test_dataset.set_format("torch")


# --------------------------------------------------
# Step 4: Load pre-trained DistilBERT model
# --------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)


# --------------------------------------------------
# Step 5: Define evaluation metrics
# --------------------------------------------------

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary"
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# --------------------------------------------------
# Step 6: Define training arguments
# --------------------------------------------------

training_args = TrainingArguments(
    output_dir="./sentiment_model",
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=100,
    report_to="none"
)


# --------------------------------------------------
# Step 7: Create Trainer
# --------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


# --------------------------------------------------
# Step 8: Train the model
# --------------------------------------------------

print("\nTraining DistilBERT...")

trainer.train()

print("\nTraining completed!")


# --------------------------------------------------
# Step 9: Evaluate the model
# --------------------------------------------------

print("\nModel Evaluation:")

results = trainer.evaluate()

for key, value in results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")


# --------------------------------------------------
# Step 10: Test custom reviews
# --------------------------------------------------

def predict_sentiment(review):

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    outputs = model(**inputs)

    prediction = np.argmax(
        outputs.logits.detach().numpy()
    )

    if prediction == 1:
        return "Positive"
    else:
        return "Negative"


reviews = [
    "This movie was absolutely fantastic and I loved every minute of it.",
    "The movie was boring and a complete waste of time.",
    "The acting was excellent and the story was very interesting.",
    "I did not enjoy this movie at all."
]

print("\nSentiment Predictions:")
print("-" * 50)

for review in reviews:
    sentiment = predict_sentiment(review)
    print(f"\nReview: {review}")
    print(f"Sentiment: {sentiment}")
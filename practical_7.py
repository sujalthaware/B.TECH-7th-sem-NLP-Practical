# NLP Lab: Text Generation using RNN

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# --------------------------------------------------
# Step 1: Training text
# --------------------------------------------------

text = """
natural language processing is a field of artificial intelligence
natural language processing helps computers understand human language
machine learning is used for natural language processing
deep learning is an important part of machine learning
python is widely used for natural language processing
"""

# Convert text to lowercase
text = text.lower()

# --------------------------------------------------
# Step 2: Create character vocabulary
# --------------------------------------------------

chars = sorted(list(set(text)))

char_to_int = {char: i for i, char in enumerate(chars)}
int_to_char = {i: char for i, char in enumerate(chars)}

print("Number of unique characters:", len(chars))
print("Characters:", chars)

# --------------------------------------------------
# Step 3: Create input sequences and target characters
# --------------------------------------------------

sequence_length = 40

X = []
y = []

for i in range(0, len(text) - sequence_length):
    sequence = text[i:i + sequence_length]
    target = text[i + sequence_length]

    X.append([char_to_int[char] for char in sequence])
    y.append(char_to_int[target])

X = np.array(X)
y = np.array(y)

print("\nNumber of training sequences:", len(X))
print("Input shape:", X.shape)

# --------------------------------------------------
# Step 4: Build RNN model
# --------------------------------------------------

model = Sequential([
    Embedding(input_dim=len(chars), output_dim=32),
    SimpleRNN(128, return_sequences=False),
    Dense(len(chars), activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------
# Step 5: Train the RNN
# --------------------------------------------------

model.fit(
    X,
    y,
    epochs=30,
    batch_size=32,
    verbose=1
)

# --------------------------------------------------
# Step 6: Generate text
# --------------------------------------------------

def generate_text(seed_text, length=200):
    generated_text = seed_text

    for _ in range(length):
        # Take the last sequence_length characters
        sequence = generated_text[-sequence_length:]

        # Convert characters to integers
        encoded_sequence = [
            char_to_int.get(char, 0)
            for char in sequence
        ]

        # Pad if the seed is shorter than sequence_length
        if len(encoded_sequence) < sequence_length:
            encoded_sequence = (
                [0] * (sequence_length - len(encoded_sequence))
                + encoded_sequence
            )

        # Convert to NumPy array
        input_sequence = np.array(
            [encoded_sequence]
        )

        # Predict next character
        prediction = model.predict(
            input_sequence,
            verbose=0
        )

        predicted_index = np.argmax(prediction[0])

        # Convert integer back to character
        next_char = int_to_char[predicted_index]

        generated_text += next_char

    return generated_text


# --------------------------------------------------
# Step 7: Generate sample text
# --------------------------------------------------

seed = "natural language processing"

print("\nGenerated Text:")
print(generate_text(seed, 200))
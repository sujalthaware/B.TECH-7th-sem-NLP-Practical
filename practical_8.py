# NLP Lab: English-to-Hindi Translation using Seq2Seq

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# Step 1: Create a small parallel corpus
# --------------------------------------------------

english_sentences = [
    "hello",
    "how are you",
    "what is your name",
    "my name is john",
    "good morning",
    "good night",
    "thank you",
    "welcome",
    "i am fine",
    "where are you",
    "i am a student",
    "i love india",
    "see you tomorrow",
    "what are you doing",
    "i am learning"
]

hindi_sentences = [
    "नमस्ते",
    "आप कैसे हैं",
    "आपका नाम क्या है",
    "मेरा नाम जॉन है",
    "सुप्रभात",
    "शुभ रात्रि",
    "धन्यवाद",
    "स्वागत है",
    "मैं ठीक हूँ",
    "आप कहाँ हैं",
    "मैं एक विद्यार्थी हूँ",
    "मुझे भारत से प्यार है",
    "कल मिलते हैं",
    "आप क्या कर रहे हैं",
    "मैं सीख रहा हूँ"
]

# Add start and end tokens to Hindi sentences
hindi_sentences = [
    "<start> " + sentence + " <end>"
    for sentence in hindi_sentences
]

# --------------------------------------------------
# Step 2: Tokenize English and Hindi
# --------------------------------------------------

eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(english_sentences)

hin_tokenizer = Tokenizer(
    filters='',
    lower=False
)
hin_tokenizer.fit_on_texts(hindi_sentences)

eng_sequences = eng_tokenizer.texts_to_sequences(english_sentences)
hin_sequences = hin_tokenizer.texts_to_sequences(hindi_sentences)

# --------------------------------------------------
# Step 3: Pad sequences
# --------------------------------------------------

max_eng_len = max(len(seq) for seq in eng_sequences)
max_hin_len = max(len(seq) for seq in hin_sequences)

encoder_input_data = pad_sequences(
    eng_sequences,
    maxlen=max_eng_len,
    padding="post"
)

decoder_sequences = pad_sequences(
    hin_sequences,
    maxlen=max_hin_len,
    padding="post"
)

# Decoder input excludes the last token
decoder_input_data = decoder_sequences[:, :-1]

# Decoder target excludes the first token
decoder_target_data = decoder_sequences[:, 1:]

# --------------------------------------------------
# Step 4: Define vocabulary sizes
# --------------------------------------------------

eng_vocab_size = len(eng_tokenizer.word_index) + 1
hin_vocab_size = len(hin_tokenizer.word_index) + 1

print("English vocabulary size:", eng_vocab_size)
print("Hindi vocabulary size:", hin_vocab_size)

# --------------------------------------------------
# Step 5: Build Encoder
# --------------------------------------------------

latent_dim = 128

encoder_inputs = Input(
    shape=(None,),
    name="encoder_input"
)

encoder_embedding = tf.keras.layers.Embedding(
    eng_vocab_size,
    latent_dim,
    mask_zero=True
)(encoder_inputs)

encoder_lstm = LSTM(
    latent_dim,
    return_state=True
)

_, state_h, state_c = encoder_lstm(
    encoder_embedding
)

encoder_states = [state_h, state_c]

# --------------------------------------------------
# Step 6: Build Decoder
# --------------------------------------------------

decoder_inputs = Input(
    shape=(None,),
    name="decoder_input"
)

decoder_embedding_layer = tf.keras.layers.Embedding(
    hin_vocab_size,
    latent_dim,
    mask_zero=True
)

decoder_embedding = decoder_embedding_layer(
    decoder_inputs
)

decoder_lstm = LSTM(
    latent_dim,
    return_sequences=True,
    return_state=True
)

decoder_outputs, _, _ = decoder_lstm(
    decoder_embedding,
    initial_state=encoder_states
)

decoder_dense = Dense(
    hin_vocab_size,
    activation="softmax"
)

decoder_outputs = decoder_dense(
    decoder_outputs
)

# --------------------------------------------------
# Step 7: Create Seq2Seq model
# --------------------------------------------------

model = Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------
# Step 8: Train the model
# --------------------------------------------------

model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=4,
    epochs=300,
    verbose=0
)

print("\nModel training completed!")

# --------------------------------------------------
# Step 9: Create inference models
# --------------------------------------------------

# Encoder inference model
encoder_model = Model(
    encoder_inputs,
    encoder_states
)

# Decoder inference inputs
decoder_state_input_h = Input(
    shape=(latent_dim,)
)

decoder_state_input_c = Input(
    shape=(latent_dim,)
)

decoder_states_inputs = [
    decoder_state_input_h,
    decoder_state_input_c
]

decoder_emb = decoder_embedding_layer(
    decoder_inputs
)

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_emb,
    initial_state=decoder_states_inputs
)

decoder_outputs = decoder_dense(
    decoder_outputs
)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs, state_h, state_c]
)

# --------------------------------------------------
# Step 10: Translation function
# --------------------------------------------------

reverse_hindi_index = {
    index: word
    for word, index in hin_tokenizer.word_index.items()
}


def translate(sentence):

    # Convert English sentence to sequence
    sequence = eng_tokenizer.texts_to_sequences([sentence])

    sequence = pad_sequences(
        sequence,
        maxlen=max_eng_len,
        padding="post"
    )

    # Encode input sentence
    states_value = encoder_model.predict(
        sequence,
        verbose=0
    )

    # Start token
    start_token = hin_tokenizer.word_index["<start>"]

    end_token = hin_tokenizer.word_index["<end>"]

    target_seq = np.array([[start_token]])

    decoded_sentence = []

    for _ in range(max_hin_len):

        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value,
            verbose=0
        )

        sampled_token_index = np.argmax(
            output_tokens[0, -1, :]
        )

        sampled_word = reverse_hindi_index.get(
            sampled_token_index,
            ""
        )

        if sampled_token_index == end_token:
            break

        if sampled_word not in ["<start>", "<end>", ""]:
            decoded_sentence.append(sampled_word)

        target_seq = np.array(
            [[sampled_token_index]]
        )

        states_value = [h, c]

    return " ".join(decoded_sentence)


# --------------------------------------------------
# Step 11: Test the translator
# --------------------------------------------------

test_sentences = [
    "hello",
    "good morning",
    "thank you",
    "how are you",
    "i am fine"
]

print("\nEnglish -> Hindi Translation")
print("-" * 40)

for sentence in test_sentences:
    translation = translate(sentence)
    print(f"{sentence} -> {translation}")
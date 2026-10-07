# NLP Lab: Rule-Based Chatbot

import re
import random

# --------------------------------------------------
# Chatbot responses
# --------------------------------------------------

responses = {
    "greeting": [
        "Hello! How can I help you?",
        "Hi! Nice to meet you.",
        "Hello! What can I do for you?"
    ],

    "name": [
        "My name is NLP Bot.",
        "I am a simple NLP-based chatbot."
    ],

    "how_are_you": [
        "I am doing great!",
        "I am fine, thank you.",
        "I'm good. How are you?"
    ],

    "thanks": [
        "You're welcome!",
        "Glad I could help!",
        "No problem!"
    ],

    "bye": [
        "Goodbye!",
        "See you later!",
        "Have a nice day!"
    ],

    "help": [
        "I can answer simple questions about myself.",
        "You can ask me about my name, capabilities, or say hello."
    ]
}


# --------------------------------------------------
# NLP preprocessing
# --------------------------------------------------

def preprocess(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize
    tokens = text.split()

    return tokens


# --------------------------------------------------
# Identify user intent
# --------------------------------------------------

def get_intent(tokens):

    # Greeting
    if any(word in tokens for word in
           ["hello", "hi", "hey", "good"]):
        return "greeting"

    # Name
    elif "name" in tokens:
        return "name"

    # How are you
    elif ("how" in tokens and "you" in tokens):
        return "how_are_you"

    # Thanks
    elif any(word in tokens for word in
             ["thanks", "thank"]):
        return "thanks"

    # Help
    elif any(word in tokens for word in
             ["help", "assist"]):
        return "help"

    # Goodbye
    elif any(word in tokens for word in
             ["bye", "goodbye", "exit", "quit"]):
        return "bye"

    else:
        return "unknown"


# --------------------------------------------------
# Chatbot function
# --------------------------------------------------

def chatbot():

    print("====================================")
    print("        NLP Rule-Based Chatbot")
    print("====================================")
    print("Type 'bye' to exit.\n")

    while True:

        user_input = input("You: ")

        # Preprocess input
        tokens = preprocess(user_input)

        # Determine intent
        intent = get_intent(tokens)

        # Exit chatbot
        if intent == "bye":
            print("Bot:", random.choice(responses["bye"]))
            break

        # Generate response
        if intent in responses:
            reply = random.choice(responses[intent])
        else:
            reply = (
                "Sorry, I don't understand that. "
                "Please try asking something else."
            )

        print("Bot:", reply)


# --------------------------------------------------
# Start chatbot
# --------------------------------------------------

chatbot()
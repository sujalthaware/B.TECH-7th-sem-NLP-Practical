# NLP Lab: Named Entity Recognition using spaCy

import spacy

# Load the pre-trained English language model
nlp = spacy.load("en_core_web_sm")

# Given text
text = """
Sundar Pichai is the CEO of Google.
He was born in Chennai, India.
Google has its headquarters in Mountain View, California.
Microsoft is another technology company founded by Bill Gates.
"""

# Process the text
doc = nlp(text)

print("Original Text:")
print(text)

print("\nNamed Entities:")
print("-" * 40)

# Display named entities
for ent in doc.ents:
    print(f"Entity: {ent.text:20} Label: {ent.label_}")

# Classify selected entity types
print("\nEntity Classification:")
print("-" * 40)

for ent in doc.ents:
    if ent.label_ == "PERSON":
        category = "Person"
    elif ent.label_ == "ORG":
        category = "Organization"
    elif ent.label_ in ["GPE", "LOC"]:
        category = "Location"
    else:
        category = ent.label_

    print(f"{ent.text:20} -> {category}")
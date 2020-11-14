import spacy

nlp = spacy.load("output/model-best")
text = ""

print("type : 'quit' to exit")

while text != "quit":
    text = input("Please enter example input: ")

    doc = nlp(text)

    print(doc.cats["positive"])
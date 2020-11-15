import spacy
# load thebest model from training
nlp = spacy.load("output/model-best")
text = ""
print("type : 'quit' to exit")
# predict the sentiment until someone writes quit
while text != "quit":
    text = input("Please enter example input: ")
    doc = nlp(text)
    if doc.cats['positive'] >.5:
        print(f"the sentiment is positive")
    else:
        print(f"the sentiment is negative")
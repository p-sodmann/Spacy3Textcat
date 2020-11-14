import spacy

# tqdm is a great progress bar for python
# tqdm.auto automatically selects a text based progress for the console 
# and html based output in jupyter notebooks
from tqdm.auto import tqdm

# DocBin is spacys new way to store Docs in a binary format for training later
from spacy.tokens import DocBin

# We want to classify movie reviews as positive or negative
from ml_datasets import imdb

# load movie reviews as a tuple (text, label)
train_data, valid_data = imdb()

# load a medium sized english language model in spacy
nlp = spacy.load("en_core_web_md")

# we are so far only interested in the first 5000 reviews
# this will keep the training time short.
# In practice take as much data as you can get.
num_texts = 5000


def make_docs(data):
    """
    this will take a list of texts and labels and transform them in spacy documents
    
    texts: List(str)
    labels: List(labels)
    
    returns: List(spacy.Doc.doc)
    """
    
    docs = []

    # nlp.pipe([texts]) is way faster than running nlp(text) for each text
    # as_tuples allows us to pass in a tuple, the first one is treated as text
    # the second one will get returned as it is.
    
    for doc, label in tqdm(nlp.pipe(data, as_tuples=True), total = len(data)):
        
        # we need to set the (text)cat(egory) for each document
        doc.cats["positive"] = label
        
        # put them into a nice list
        docs.append(doc)
    
    return docs


# we are so far only interested in the first 5000 reviews
# this will keep the training time short.
# In practice take as much data as you can get.
# you can always reduce it to make the script even faster.
num_texts = 5000


# first we need to transform all the training data
train_docs = make_docs(train_data[:num_texts])
# then we save it in a binary file to disc
doc_bin = DocBin(docs=train_docs)
doc_bin.to_disk("./data/train.spacy")

# repeat for validation data
valid_docs = make_docs(valid_data[:num_texts])
doc_bin = DocBin(docs=valid_docs)
doc_bin.to_disk("./data/valid.spacy")
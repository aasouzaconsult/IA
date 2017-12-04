import numpy as np
import pickle
import operator
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from nltk import word_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

def preprocess(text):
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()]
    return doc
k = 3

ng20 = fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes'))
texts, y = ng20.data, ng20.target
count = len(texts)
document_term = TfidfVectorizer()
matrix_document_term = document_term.fit_transform(texts).toarray()

matrix_document_document = np.dot(matrix_document_term, np.transpose(matrix_document_term))

print matrix_document_document.shape
with open('matrix_document_document', 'wb') as output:
    pickle.dump(matrix_document_document, output, pickle.HIGHEST_PROTOCOL)

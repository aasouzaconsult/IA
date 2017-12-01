############
## Inicio ##
############

import nltk
import gensim
import logging
import matplotlib.pyplot as plt
import numpy as np
import sys
from time import time
from gensim import utils
from gensim.models import Word2Vec
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
from matplotlib import pyplot

# Logar operações
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

download('punkt') #tokenizer, run once
download('stopwords') #stopwords dictionary, run once
stop_words = stopwords.words('english')

def preprocess(text):
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()] #restricts string to alphabetic characters only
    return doc

print("Loading dataset...")
t0 = time()

print("Importing collection 20NewsGroups...")
# Usando o 20 NewsGroups
ng20 = fetch_20newsgroups(subset='all',
                          remove=('headers', 'footers', 'quotes'))

# text and ground truth labels
texts, y = ng20.data, ng20.target

print("Preprocessing...")
sentences = [preprocess(text) for text in texts]

print("Training the model...")
# Treinando o Modelo
model = Word2Vec(sentences, min_count=1, size = 278)
# size = Raiz quadrada do tamanho do vocabulário (uma heurística)

print("done in %0.3fs." % (time() - t0))

# summarize the loaded model
print(model)

# summarize vocabulary
words = list(model.wv.vocab)
print(words)

# access vector for one word
print(model['sentence'])

# save model
model.save('model.bin')
model.wv.save_word2vec_format('model.txt', binary=False) # Em texto

# load model
new_model = Word2Vec.load('model.bin')
print(new_model)

# Vetores do modelo treinado
X = model[model.wv.vocab]

# Testando similaridade
model.similarity('sentence','darrylo')

# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

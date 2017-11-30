import nltk
import gensim
from gensim import utils
import numpy as np
import sys
from sklearn.datasets import fetch_20newsgroups
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

download('punkt') #tokenizer, run once
download('stopwords') #stopwords dictionary, run once
stop_words = stopwords.words('english')

def preprocess(text):
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()] #restricts string to alphabetic characters only
    return doc

# Usando o 20 NewsGroups
ng20 = fetch_20newsgroups(subset='all',
                          remove=('headers', 'footers', 'quotes'))

# text and ground truth labels
texts, y = ng20.data, ng20.target

sentences = [preprocess(text) for text in texts]

# Treinando o Modelo
model = Word2Vec(sentences, min_count=1)

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
model.similarity('sentence','more')
model.similarity('more','sentence')
model.similarity('and','for')
model.similarity('sentence','this')
model.most_similar('yet')

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

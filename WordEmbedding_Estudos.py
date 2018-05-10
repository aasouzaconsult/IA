#########################################
# Exemplo 1 - Word2Vec - GESIM + Google #
#########################################
import nltk
import gensim
from gensim import utils
import numpy as np
import sys
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import metrics

# Importando modelo do google
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

download('punkt') #tokenizer, run once
download('stopwords') #stopwords dictionary, run once
stop_words = stopwords.words('english')

def preprocess(text):
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()] #restricts string to alphabetic characters only
    return doc

ng20 = fetch_20newsgroups(subset='all',
                          remove=('headers', 'footers', 'quotes'))

ng20_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes')) # train
ng20_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes')) # test

# text and ground truth labels
#texts, y = ng20.data, ng20.target
texts, y = ng20_train.data, ng20_train.target # train
texts_test, y_test = ng20_test.data, ng20_test.target # test

corpus = [preprocess(text) for text in texts] # train
corpus_test = [preprocess(text_test) for text_test in texts_test] # test

def filter_docs(corpus, texts, labels, condition_on_doc):
    """
    Filter corpus, texts and labels given the function condition_on_doc which takes
    a doc.
    The document doc is kept if condition_on_doc(doc) is true.
    """
    number_of_docs = len(corpus)

    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if condition_on_doc(doc)]

    labels = [i for (i, doc) in zip(labels, corpus) if condition_on_doc(doc)]
    corpus = [doc for doc in corpus if condition_on_doc(doc)]

    print("{} docs removed".format(number_of_docs - len(corpus)))

    return (corpus, texts, labels)

# corpus - Lista de palavras
# texts  - texto puro
corpus, texts, y = filter_docs(corpus, texts, y, lambda doc: (len(doc) != 0)) # train
corpus_test, texts_test, y_test = filter_docs(corpus_test, texts_test, y_test, lambda doc: (len(doc) != 0)) # test

### Remove OOV words and documents with no words in model dictionary
def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.vocab] # Palavras no Word2Vec para cada Documento
    return np.mean(word2vec_model[doc], axis=0) # Média de todas as palavras(vetor 300dim) do documento (por documento)

def has_vector_representation(word2vec_model, doc):
    """check if at least one word of the document is in the word2vec dictionary"""
    return not all(word not in word2vec_model.vocab for word in doc)

corpus, texts, y = filter_docs(corpus, texts, y, lambda doc: has_vector_representation(model, doc)) # train
corpus_test, texts_test, y_test = filter_docs(corpus_test, texts_test, y_test, lambda doc: has_vector_representation(model, doc)) # train

x =[]
for doc in corpus: #look up each doc in model
    x.append(document_vector(model, doc))

X = np.array(x) #list to array

# test
x_test =[]
for doc in corpus_test: #look up each doc in model
    x_test.append(document_vector(model, doc))

X_test = np.array(x_test) #list to array
# ---------------------------------

np.save('documents_vectors.npy', X)  #np.savetxt('documents_vectors.txt', X)
np.save('labels.npy', y)             #np.savetxt('labels.txt', y)

X.shape, len(y)

X[1]
X_test[1]

# Testing 
# -----------------------------------------------------
for doc in corpus: #look up each doc in model
    doc = [word for word in doc if word in model.vocab]
    print doc
    print np.mean(word2vec_model[doc], axis=0)

svm = OneVsRestClassifier(SVC(kernel="linear", verbose=True)) # Modelo
svm.fit(X, y) # Treinamento
pred_svm = svm.predict(X_test) # predizendo
print(metrics.classification_report(y_test, pred_svm)) # por Categoria

#              precision    recall  f1-score   support
# avg / total       0.61      0.61      0.60      7291

# -----------------------------------------------------

### Sanity check
texts[4664]

y[4664], ng20.target_names[11]

### Plot 2 PCA components
pca = PCA(n_components=2)
x_pca = pca.fit_transform(X)

plt.figure(1, figsize=(30, 20),)
plt.scatter(x_pca[:, 0], x_pca[:, 1],s=100, c=y, alpha=0.2)

### Plot t-SNE (Lento)
# from sklearn.manifold import TSNE
# X_tsne = TSNE(n_components=2, verbose=2).fit_transform(X)

# plt.figure(1, figsize=(30, 20),)
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1],s=100, c=y, alpha=0.2)
'''
[t-SNE] Computing pairwise distances...
[t-SNE] Computing 91 nearest neighbors...
[t-SNE] Computed conditional probabilities for sample 1000 / 18282
[t-SNE] Computed conditional probabilities for sample 2000 / 18282
[t-SNE] Computed conditional probabilities for sample 3000 / 18282
[t-SNE] Computed conditional probabilities for sample 4000 / 18282
[t-SNE] Computed conditional probabilities for sample 5000 / 18282
[t-SNE] Computed conditional probabilities for sample 6000 / 18282
[t-SNE] Computed conditional probabilities for sample 7000 / 18282
[t-SNE] Computed conditional probabilities for sample 8000 / 18282
[t-SNE] Computed conditional probabilities for sample 9000 / 18282
[t-SNE] Computed conditional probabilities for sample 10000 / 18282
[t-SNE] Computed conditional probabilities for sample 11000 / 18282
[t-SNE] Computed conditional probabilities for sample 12000 / 18282
[t-SNE] Computed conditional probabilities for sample 13000 / 18282
[t-SNE] Computed conditional probabilities for sample 14000 / 18282
[t-SNE] Computed conditional probabilities for sample 15000 / 18282
[t-SNE] Computed conditional probabilities for sample 16000 / 18282
[t-SNE] Computed conditional probabilities for sample 17000 / 18282
[t-SNE] Computed conditional probabilities for sample 18000 / 18282
[t-SNE] Computed conditional probabilities for sample 18282 / 18282
[t-SNE] Mean sigma: 0.146211
[t-SNE] Iteration 25: error = 0.6600548, gradient norm = 0.0000475
[t-SNE] Iteration 25: gradient norm 0.000048. Finished.
[t-SNE] Iteration 50: error = 0.6595663, gradient norm = 0.0056467
[t-SNE] Iteration 75: error = 0.6572590, gradient norm = 0.0040334
[t-SNE] Iteration 100: error = 0.6565863, gradient norm = 0.0036505
[t-SNE] Error after 100 iterations with early exaggeration: 0.656586
[t-SNE] Iteration 125: error = 0.6521526, gradient norm = 0.0023158
[t-SNE] Iteration 150: error = 0.6506326, gradient norm = 0.0020567
[t-SNE] Iteration 175: error = 0.6502423, gradient norm = 0.0020017
[t-SNE] Iteration 200: error = 0.6501332, gradient norm = 0.0019869
[t-SNE] Iteration 225: error = 0.6501032, gradient norm = 0.0019828
[t-SNE] Iteration 250: error = 0.6500965, gradient norm = 0.0019814
[t-SNE] Iteration 275: error = 0.6500937, gradient norm = 0.0019810
[t-SNE] Iteration 300: error = 0.6500929, gradient norm = 0.0019810
[t-SNE] Iteration 300: error difference 0.000000. Finished.
[t-SNE] Error after 300 iterations: 0.650093
Out[322]: 
<matplotlib.collections.PathCollection at 0x1885f6eb8>
<matplotlib.figure.Figure at 0x1884e9400>
'''

#################################
# Exemplo 2 - Word2Vec - Google #
#################################
#https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/
#detalhar tf-idf (exemplo na pagina acima)
#pip install --trusted-host pypi.python.org gensim

from gensim.models import Word2Vec

#loading the downloaded model
model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, norm_only=True)

#the model is loaded. It can be used to perform all of the tasks mentioned above.

# getting word vectors of a word
dog = model['dog']

#performing king queen magic (slow)
print(model.most_similar(positive=['woman', 'king'], negative=['man']))
#[(u'queen', 0.7118192315101624), 
# (u'monarch', 0.6189674735069275), 
# (u'princess', 0.5902431607246399), 
# (u'crown_prince', 0.5499460697174072), 
# (u'prince', 0.5377321243286133), 
# (u'kings', 0.5236844420433044), 
# (u'Queen_Consort', 0.5235946178436279), 
# (u'queens', 0.5181134343147278), 
# (u'sultan', 0.5098593235015869), 
# (u'monarchy', 0.5087411403656006)]

#picking odd one out
print(model.doesnt_match("breakfast cereal dinner lunch".split()))

#printing similarity index
print(model.similarity('woman', 'man'))

###############################
# Exemplo 3 - Word2Vec - NLTK #
###############################
# https://streamhacker.com/2014/12/29/word2vec-nltk/
from gensim.models import Word2Vec
from nltk.corpus import brown, movie_reviews, treebank
b = Word2Vec(brown.sents())
b.most_similar('money', topn=5)

# Referências
# https://github.com/sdimi/average-word2vec/blob/master/avg_word2vec_from_documents.py
# https://www.kernix.com/blog/similarity-measure-of-textual-documents_p12
# https://www.analyticsvidhya.com/blog/2017/10/essential-nlp-guide-data-scientists-top-10-nlp-tasks/

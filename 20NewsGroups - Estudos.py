# http://scikit-learn.org/stable/datasets/twenty_newsgroups.html
# http://qwone.com/~jason/20Newsgroups/

##########################
# Conjuntos e Categorias #
##########################

from sklearn.datasets import fetch_20newsgroups
from time import time

print("Loading dataset...")
t0 = time()
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
print("done in %0.3fs." % (time() - t0))

from pprint import pprint
pprint(list(newsgroups_train.target_names))

newsgroups_train.filenames.shape

###########################################
# Colocando em um vetor - Só 4 categorias #
###########################################

from sklearn.feature_extraction.text import TfidfVectorizer
categories = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
vectors.shape

####################
# Exemplo Completo #
####################

#http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

# Só informações das 4 categorias acima
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_train.target_names #ver as categorias

len(twenty_train.data)
len(twenty_train.filenames)

# verificando a primeira linha
print("\n".join(twenty_train.data[0].split("\n")[:]))
print(twenty_train.target_names[twenty_train.target[0]])

# Categorias por número - mais fácil para os modelos
twenty_train.target[0] # categoria do exemplo acima
twenty_train.target[:10]

# Loop para ver os nomes das categorias
for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])

# Para realizar a aprendizagem de máquina em documentos de texto, primeiro precisamos transformar o conteúdo do texto
# em vetores de características numéricas.
# Bag of Words - vetores Esparsos

# Text preprocessing, tokenizing and filtering of stopwords are included in a high level component that is able to build
#a dictionary of features and transform documents to feature vectors:
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape
X_train_counts.data
# CountVectorizer supports counts of N-grams of words or consecutive characters. Once fitted, the vectorizer has built
# a dictionary of feature indices:
count_vect.vocabulary_.get(u'algorithm') # indice da palavras algorithm
# O valor do índice de uma palavra no vocabulário está ligado à sua frequência em todo o corpus de treinamento.

# ver o vocabulário
count_vect.vocabulary_

# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
X_train_tfidf.data # ver alguns dados

##############################
# Training a classifier - NB #
##############################
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target) # Treinamento o modelo

# Teste
docs_new = ['God is love', 'OpenGL on the GPU is fast'] # Novas entradas
# Observar que aqui é só transform ao invés do fit.transform usado no treinamento
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf  = tfidf_transformer.transform(X_new_counts)

# Prevendo (com base no treinamento)
predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

# Pipeline
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])
text_clf.fit(twenty_train.data, twenty_train.target)

# Evaluation of the performance on the test set¶
import numpy as np
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)  # Usando o Pipeline com Naive Bayes
np.mean(predicted == twenty_test.target) # Média de acertos
# Acurácia de 0.83%

# Vamos ver se melhora com SVM
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42)),])
text_clf.fit(twenty_train.data, twenty_train.target)

predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)
# Acurácia de 0.91%

# Mais métricas
from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,
                                    target_names=twenty_test.target_names))
metrics.confusion_matrix(twenty_test.target, predicted)

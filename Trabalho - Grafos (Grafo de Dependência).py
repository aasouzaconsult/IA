# Base: NPL - National Physical Laboratory ######
# 11429 - Documentos | 93 - Querys | 13 - Termos - Query 81 | 84 - Documentos - Query 41 | Termos - 7878

# Auxiliares
# http://www.nltk.org/book/ch07.html
# http://nishutayaltech.blogspot.com.br/2015/02/penn-treebank-pos-tags-in-natural.html
# http://www.cs.cornell.edu/courses/cs474/2004fa/lec1.pdf (tipos)
# https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk
# https://www.clips.uantwerpen.be/pages/pattern-en
# Natural Language Processing - http://www.ling.helsinki.fi/kit/2008s/clt231/nltk-0.9.5/doc/en/book.html

import nltk
import string
import numpy as np
import pickle
import operator
import matplotlib.pyplot as pl
import sys
from nltk import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from time import time

text_trans = []
graf_trans = []
stemmer    = PorterStemmer()

# Gramática
# grammar = "NP: {<JJ>?<NP>?<VBG>*<NN>}" # Exemplo
# Creio que essa gramática abaixo seja boa
grammar = """
	NP:   {<PRP>?<JJ.*>*<NN.*>+}
	CP:   {<JJR|JJS>}
	VERB: {<VB.*>}
	THAN: {<IN>}
	COMP: {<DT>?<NP><RB>?<VERB><DT>?<CP><THAN><DT>?<NP>}
	"""

# NP - noun phrase ( their public lectures)
# PRP - pronoun (they)
# JJ - adjective (public)
# NN - singular noun (pyramid)
# VB - base (He may like/VB cookies)
# IN - preposition (in)
	
def tokenize_stopwords_stemmer(text, stemmer, query):
    no_punctuation = text.translate(None, string.punctuation)
    tokens = nltk.word_tokenize(no_punctuation)
    text_filter = [w for w in tokens if not w in stopwords.words('english')]
    text_final = ''
    if query == True: # Se for query
        for k in range(0, len(text_filter)):
            for i in wn.synsets(text_filter[k]): # Uso do WordNet
                for s in i.lemma_names():
                    text_filter.append(s)

    for k in range(0, len(text_filter)):
       #text_final +=str(stemmer.stem(text_filter[k]))
        text_final += str(text_filter[k])
        if k != len(text_filter)-1:
            text_final+=" "
            pass
    return text_final

# Processo de geração das matrizes - DT (Documento-Termo), TT (Termo-Termo) e a Matriz de Termos
def save_object(obj, filename):
    with open('objects/'+filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def generate_matrix():
    document_term = CountVectorizer()
    # Salvando em arquivo
    matrix_document_term = document_term.fit_transform(text_trans)
    save_object(document_term.get_feature_names(), 'terms_npl.dt')
    matrix_dt = np.matrix(matrix_document_term.toarray())
    save_object(matrix_dt, 'matrix_npl.dt')
    matrix_tt = np.dot(np.transpose(matrix_dt), matrix_dt)
    save_object(matrix_tt, 'matrix_npl.tt')
    pass

def organizes_documents():
    files = open('npl/doc-text', 'r').read().split('/')
    for i in range(0,len(files[:10])):
        text = files[i].strip()
        text = text.replace(str(i+1), '')
        text = text.strip()
        text_trans.append(tokenize_stopwords_stemmer(text.lower(), stemmer, False))
    generate_matrix()

# Executando
# Sem radicalização (stemmer)
organizes_documents()

# Carregar as matrizes em variáveis (Não utilizado por enquanto)
matrix_dt           = load_object('objects/matrix_npl.dt')
matrix_tt           = load_object('objects/matrix_npl.tt')
terms_dt            = load_object('objects/terms_npl.dt')

# Visualizando os dados
text_trans
text_trans[1]

# Grafo de Dependência (Teste)
text     = word_tokenize(text_trans[1])
sentence = nltk.pos_tag(text)
cp       = nltk.RegexpParser(grammar)
result   = cp.parse(sentence)
result.draw()

# Todos
for i in range(0, len(text_trans)):
    graf     = word_tokenize(text_trans[i])
    sentence = nltk.pos_tag(graf)
    cp       = nltk.RegexpParser(grammar)
    result   = cp.parse(sentence)
    graf_trans.append(result)


'''
#################
# 20 NewsGroups #
#################
# http://scikit-learn.org/stable/datasets/twenty_newsgroups.html
# http://qwone.com/~jason/20Newsgroups/

##########################
# Conjuntos e Categorias #
##########################
import nltk
from nltk import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint

# categories = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
categories = ['comp.graphics']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test  = fetch_20newsgroups(subset='test', categories=categories)

# verificar categorias
pprint(list(newsgroups_train.target_names))

# Numero de Registros
# newsgroups_train.filenames.shape
len(newsgroups_train.data)

# Vendo uma linha
newsgroups_train.data[1]
newsgroups_train.data[2]
print("\n".join(newsgroups_train.data[1].split("\n")[:]))
print("\n".join(newsgroups_train.data[2].split("\n")[:]))
print(newsgroups_train.target_names[newsgroups_train.target[1]])

# Tratando o texto

# Alguns testes
text = word_tokenize("I wonder how many atheists out there care to speculateon the face of the world.")
sentence = nltk.pos_tag(text)
grammar = "NP: {<DT>?<JJ>*<NN>}" # Exemplo - tem que ser criada uma específica
cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
result.draw()
'''

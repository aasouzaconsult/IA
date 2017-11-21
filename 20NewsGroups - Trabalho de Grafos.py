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
print("\n".join(newsgroups_train.data[1].split("\n")[:]))
print(newsgroups_train.target_names[newsgroups_train.target[1]])

# Tratando o texto


# Alguns testes
text = word_tokenize("I wonder how many atheists out there care to speculateon the face of the world.")
sentence = nltk.pos_tag(text)

grammar = "NP: {<DT>?<JJ>*<NN>}" # Exemplo - tem que ser criada uma específica

cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
result.draw()

#########
# Livro #
#########
# http://www.ling.helsinki.fi/kit/2008s/clt231/nltk-0.9.5/doc/en/book.html

# Dataset
txtAlex = "Information Retrieval for retrieval of text automatic", "Study more about information retrieval", "information is to all", "text is very important for all"
txtAlex

#####################
# Documento - Termo #
#####################

from sklearn.feature_extraction.text import CountVectorizer
txtAlex_vect = CountVectorizer()
txtAlex_train = txtAlex_vect.fit_transform(txtAlex)

# Visualizando os dados
txtAlex_train.shape
txtAlex_train.data
txtAlex_train[1]

# Vocabulário
txtAlex_vect.vocabulary_

##########
# TF-IDF #
##########

from sklearn.feature_extraction.text import TfidfTransformer
txtAlex_tfidf = TfidfTransformer()
# Ver parametros
txtAlex_tfidf._get_param_names()

txtAlex_train_tfidf = txtAlex_tfidf.fit_transform(txtAlex_train)
txtAlex_train_tfidf.shape
txtAlex_train_tfidf.data 

txtAlex_train_tfidf[0].data # ver dados do primeiro "documento"

# Visualizar o Array
txtAlex_vect.vocabulary_
txtAlex_train_tfidf.toarray()

############
# Word2Vec #
############

from sklearn.decomposition import PCA
from matplotlib import pyplot
from gensim.models import Word2Vec
# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]

# train model
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
model.wv.save_word2vec_format('model.txt', binary=False)

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

# https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
# https://quomodocumque.wordpress.com/2016/01/15/messing-around-with-word2vec/
# https://rare-technologies.com/word2vec-tutorial/
# https://radimrehurek.com/gensim/models/word2vec.html
# https://radimrehurek.com/gensim/scripts/word2vec_standalone.html
# https://codesachin.wordpress.com/2015/10/09/generating-a-word2vec-model-from-a-block-of-text-using-gensim-python/

#####################
# Janela deslizante #
#####################
sent = ['The', 'dog', 'gave', 'John', 'the', 'newspaper']
n = 3
[sent[i:i+n] for i in range(len(sent)-n+1)]

##########
# N-Gram #
##########
# http://www.ling.helsinki.fi/kit/2008s/clt231/nltk-0.9.5/doc/en/book.html#n_gram_tagger_index_term
# http://tetration.xyz/Ngram-Tutorial/

# https://programminghistorian.org/lessons/keywords-in-context-using-n-grams
import obo

wordstring = 'it was the best of times it was the worst of times '
wordstring += 'it was the age of wisdom it was the age of foolishness'

allMyWords = wordstring.split()
print(obo.getNGrams(allMyWords, 3))

# Detalhes
wordfreq = []
for w in allMyWords:
    wordfreq.append(allMyWords.count(w))

print("-> DETALHES <-")
print("String\n" + wordstring +"\n")
print("List\n" + str(allMyWords) + "\n")
print("Frequencies\n" + str(allMyWords) + "\n")
print("Pairs\n" + str(zip(allMyWords, wordfreq)))

#############################
# Separa e junta caracteres #
#############################
# http://www.pitt.edu/~naraehan/python2/split_join.html
mary = 'Mary had a little lamb'
mwords = mary.split() 
 ' '.join(mwords)
    
#####################################################
# Output Data as an HTML File with Python (wrapper) #
#####################################################
# https://programminghistorian.org/lessons/output-data-as-html-file

#########
# LINKS #
#########
# https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
# http://apprize.info/python/six/5.html

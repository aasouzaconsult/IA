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
## https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730

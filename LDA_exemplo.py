# Modelagem por Tópicos
# Latent Dirichlet Allocation (LDA)
# Adaptação do código: https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html

# Latent Dirichlet allocation (LDA) is a topic model that generates topics based on word frequency from a set of documents. LDA is particularly useful for finding reasonably accurate mixtures of topics within a given document set.

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = stopwords.words('english')

# Create p_stemmer of class PorterStemmer
p_stemmer   = PorterStemmer()
    
# create sample documents
doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health." 
doc_f = "My brother goes to school but does not know how to drive, who takes him is our mother."

# compile sample documents into a list
doc_set    = [doc_a, doc_b, doc_c, doc_d, doc_e, doc_f]

# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
	#print(tokens)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
	#print(stopped_tokens)
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
	#print(stemmed_tokens)
    
    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
print "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
print "Termos : Codigo ->" + str(dictionary.token2id) # Termo associado a um numero

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts] # Convert Dicionario em um saco de palavras
print "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
print "Termo(Codigo), Frequencia" + str(corpus)       # (Termo, Frequencia)
print "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
print(" ")
# print(corpus[0]) # Individualmente (Termo, Frequencia)

# Gerando o Modelo LDA - 3 Tópicos
print("################################################")
print("############ 3 Tópicos e 3 Palavras ############")
print("################################################")
print(" ")
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=3, num_words=3))
print(" ")
# Outro, agora com 2 tópicos
print("################################################")
print("############ 2 Tópicos e 3 Palavras ############")
print("################################################")
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=20)
print(" ")
print(ldamodel.print_topics(num_topics=2, num_words=3))
print(" ")

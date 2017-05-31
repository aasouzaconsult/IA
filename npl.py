######################################################
###### Base: NPL - National Physical Laboratory ######
######################################################
# 11429 - Documentos
# 93 - Querys
# 13 - Termos - Query 81
# 84 - Documentos - Query 41
# Termos - 7878

################
# Observações  #
################
# - Aula 31.05

# Matriz de Frequencia e normalizar
# BM25 
# OR  com Ranqueamento + um corte

import nltk
import string
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

stemmer = PorterStemmer()
querys = []
expansion = 5
text_trans = []

def retrieval(terms,matrix_dt):
    result_docs = []
    for term in terms:
        result_docs = result_docs + np.where(matrix_dt[:,term]>0)[0].tolist()[0]
    return set(result_docs)

def tokenize_stopwords_stemmer(text, stemmer):
    no_punctuation = text.translate(None, string.punctuation)
    tokens = nltk.word_tokenize(no_punctuation)
    text_filter = [w for w in tokens if not w in stopwords.words('english')]
    text_final = ''
    for k in range(0, len(text_filter)):
        text_final +=str(stemmer.stem(text_filter[k]))
        if k != len(text_filter)-1:
            text_final+=" "
            pass
    return text_final

def organizes_documents():
    files = open('npl/doc-text', 'r').read().split('/')
    for i in range(0,len(files)):
        text = files[i].strip()
        text = text.replace(str(i+1), '')
        text = text.strip()
        text_trans.append(tokenize_stopwords_stemmer(text.lower(), stemmer))
    generate_matrix()
	
def organizes_querys():
    files = open('npl/query-text', 'r').read().split('/')
    for i in range(0,len(files)):
        textq = files[i].strip()
        textq = textq.replace(str(i+1), '')
        textq = textq.strip()
        querys.append(textq)
			
def save_object(obj, filename):
    with open('objects/'+filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def generate_matrix():
    document_term = CountVectorizer()

    # TfIdf   = document_term.fit_transform(text_trans) # Documento X Termo
    # Td      = (TfIdf != 0).astype(int)                # Documento X Termo (0 ou 1)
    # Termos  = document_term.get_feature_names()       # Termos
    # 
    # M_TfIdf = TfIdf.transpose() # Termo X Documento
    # M_Td    = Td.transpose() 	# Termo X Documento
	# 
    # # Matriz de Termo-documento (Termo - Termo)
    # XT = np.dot(M_Td, M_Td.transpose())
    # 
    # # Matriz de Termo-documento (Documento - Documento)
    # XD = np.dot(M_Td.transpose(), M_Td)
    # 
    # # Matriz de TF-IDF (Termo - Termo)
    # KT = np.dot(M_TfIdf, M_TfIdf.transpose())
    # 
    # # Matriz de TF-IDF (Documento - Documento)
    # KD = np.dot(M_TfIdf.transpose(), M_TfIdf)

    # Salvando em arquivo
    # matrix_document_term = document_term.fit_transform(text_trans)
    matrix_document_term = document_term.fit_transform(text_trans[0:5000]) # Diminuindo para validacao
    save_object(document_term.get_feature_names(), 'terms_npl.dt')
    matrix_dt = np.matrix(matrix_document_term.toarray())
    save_object(matrix_dt, 'matrix_npl.dt')
    matrix_tt = np.dot(np.transpose(matrix_dt), matrix_dt)
    save_object(matrix_tt, 'matrix_npl.tt')
    pass
	
# Expandindo a consulta (5)
def search_expanded(query, terms_dt, matrix_tt):
    terms = []
    for i in query:
        if i in terms_dt:
            key = terms_dt.index(i)
            terms_recommended = np.sort(matrix_tt[key])[:, len(matrix_tt)-expansion:len(matrix_tt)]
            for j in terms_recommended.tolist()[0]:
                terms.append(matrix_tt[key, :].tolist()[0].index(j))
            pass
        pass
    pass
    return terms

# Consulta normal
def search (query, terms_dt, matrix_tt):
    termss = []
    for i in query:
        if i in terms_dt:
            key = terms_dt.index(i)
            termsO = np.matrix(matrix_tt[key,key])
            for j in termsO.tolist()[0]:
                termss.append(matrix_tt[key, :].tolist()[0].index(j))
            pass
        pass
    pass
    return termss
termss = search(query_token.split(' '), terms_dt, matrix_tt)

# Documentos relevantes
relevantes = []
def relevants_documents():
    relevants_resume = dict()
    files = open('npl/rlv-ass', 'r').read().split('/')
    for i in range(0,len(files)):
        textr = files[i].strip()
        textr = textr.strip()
        textr = textr.replace('\n', ' ')
        textr = textr.replace('  ', ' ')
        textr = textr.replace('  ', ' ')

        line = np.array(textr.split(' ')).tolist()
        key = int(line[0])
        for j in range(len(line)-1):
            if key in relevants_resume:
                relevants_resume[key].append(int(line[j+1]))
            else:
                relevants_resume[key] = [int(line[j+1])]
            pass
    pass
    return relevants_resume

def main():
    #organizes_documents()
    organizes_querys()
    matrix_dt = load_object('objects/matrix_npl.dt')
    matrix_tt = load_object('objects/matrix_npl.tt')
    terms_dt = load_object('objects/terms_npl.dt')
    
    for i in xrange(0,len(querys)):
        query_token          = tokenize_stopwords_stemmer(querys[i], stemmer)
        terms                = search_expanded(query_token.split(' '), terms_dt, matrix_tt)
		termss               = search(query_token.split(' '), terms_dt, matrix_tt)
	    #terms               = search_expanded(query_token.split(' '), Termos, KT)
        documents_retrieval  = retrieval(terms, matrix_dt)
	    #documents_retrieval = retrieval(terms, TfIdf)
        documents_relevants  = relevants_documents()[i+1]
        precision            = float(len(documents_retrieval.intersection(documents_relevants))) /  float(len(documents_retrieval))
        recall               = float(len(documents_retrieval.intersection(documents_relevants))) / float(len(documents_relevants))
		print "Query....: " + str(i+1)
        print "Precision: " + str(round(precision, 2)*100)
        print "Recall...: " + str(round(recall, 2)*100)
        print "############################################"
        pass
    pass

main()

######################################################
###### Base: NPL - National Physical Laboratory ######
######################################################
# 11429 - Documentos
# 93 - Querys
# 13 - Termos - Query 81
# 84 - Documentos - Query 41
# Termos - 7878

import nltk
import string
import numpy as np
import pickle
import operator
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

stemmer    = PorterStemmer()
querys     = []
querysOri  = []
expansion  = 0
expansion2 = 5
text_trans = []

#---------------------------------------------------------------------------------------#
def retrieval(terms,matrix_dt, terms_dt, query):
    result_docs = []
    for term in terms:
        sum_vector = np.sum(matrix_dt[:,term])
        norm = dict()
        for i in np.where(matrix_dt[:,term]>1)[0].tolist():
            norm[i] = float(matrix_dt[i, term])/float(sum_vector)
        norm_sort = sorted(norm.items(), key=operator.itemgetter(1),reverse=True)
        sum_norm_sort = 0
        for i in norm_sort:
            sum_norm_sort = sum_norm_sort + i[1]
            result_docs.append(i[0])
            if sum_norm_sort >= 0.5:#float(len(query))/float(len(terms)):
                break
            pass
    return set(result_docs)

def retrieval1(terms,matrix_dt):
    result_docs = []
    for term in terms:
        result_docs = result_docs + np.where(matrix_dt[:,term]>0)[0].tolist() # Coluna (varre a coluna do termo e pega os documentos)
    return set(result_docs) # set - Distinct 

#---------------------------------------------------------------------------------------#
def tokenize_stopwords_stemmer(text, stemmer, query):
    no_punctuation = text.translate(None, string.punctuation)
    tokens = nltk.word_tokenize(no_punctuation)
    text_filter = [w for w in tokens if not w in stopwords.words('english')]
    text_final = ''
    if query == True: # Se for query
        for k in range(0, len(text_filter)):
            for i in wn.synsets(text_filter[k]):
                for s in i.lemma_names():
                    text_filter.append(s)

    for k in range(0, len(text_filter)):
        text_final +=str(stemmer.stem(text_filter[k]))
        if k != len(text_filter)-1:
            text_final+=" "
            pass
    return text_final

def tokenize_stopwords_stemmer1(text, stemmer):
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

#---------------------------------------------------------------------------------------#
def organizes_documents():
    files = open('npl/doc-text', 'r').read().split('/')
    for i in range(0,len(files)):
        text = files[i].strip()
        text = text.replace(str(i+1), '')
        text = text.strip()
        text_trans.append(tokenize_stopwords_stemmer(text.lower(), stemmer, False))
    generate_matrix()

#---------------------------------------------------------------------------------------#
def organizes_querys():
    files = open('npl/query-text', 'r').read().split('/')
    for i in range(0,len(files)):
        textq = files[i].strip()
        textq = textq.replace(str(i+1), '')
        textq = textq.strip()
        querys.append(textq.lower())

#---------------------------------------------------------------------------------------#
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
    # matrix_document_term = document_term.fit_transform(text_trans[0:5000]) # Diminuindo para validacao
    save_object(document_term.get_feature_names(), 'terms_npl.dt')
    matrix_dt = np.matrix(matrix_document_term.toarray())
    save_object(matrix_dt, 'matrix_npl.dt')
    matrix_tt = np.dot(np.transpose(matrix_dt), matrix_dt)
    save_object(matrix_tt, 'matrix_npl.tt')
    pass

#---------------------------------------------------------------------------------------#
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

def search_expanded(query, terms_dt, matrix_tt):
    terms = []
    for i in query:
        if i in terms_dt:#se botar nada a ver
            key = terms_dt.index(i) # Pega a posicao que o termo se encontra
            terms_recommended = np.sort(matrix_tt[key])[:, len(matrix_tt)-expansion:len(matrix_tt)]  # Final da Linha 400 os 5 ultimos colunas (maiores)
            for j in terms_recommended.tolist()[0]:
                terms
                terms.append(matrix_tt[key, :].tolist()[0].index(j)) # Retorna o indice da frequencia j
            pass
            if key in terms == False or expansion == 0:
                terms.append(key)
        pass
    pass
    return set(terms)
# Exemplo de consulta: Minha (400) Primeira (200) Consulta (2)

# Expandir a consulta
def search_expanded2(query, terms_dt, matrix_tt):
    terms = []
    for i in query:
        if i in terms_dt:#se botar nada a ver
            key = terms_dt.index(i)
            terms_recommended = np.sort(matrix_tt[key])[:, len(matrix_tt)-expansion2:len(matrix_tt)]
            for j in terms_recommended.tolist()[0]:
                terms
                terms.append(matrix_tt[key, :].tolist()[0].index(j))
            pass
            if key in terms == False or expansion2 == 0:
                terms.append(key)
        pass
    pass
    return set(terms)

#---------------------------------------------------------------------------------------#
# Documentos relevantes
relevantes = []
def relevants_documents():
    relevants_resume = dict() # Vetor chave valor
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

#---------------------------------------------------------------------------------------#
def main():
    organizes_documents()
    organizes_querys()
    matrix_dt = load_object('objects/matrix_npl.dt')
    matrix_tt = load_object('objects/matrix_npl.tt')
    terms_dt  = load_object('objects/terms_npl.dt')

    amount_documents = len(matrix_dt)
    mean_precision   = 0
    mean_recall      = 0
    mean_acuracy     = 0
    mean_precision1  = 0
    mean_recall1     = 0
    mean_acuracy1    = 0
	
    print "############################################"
    for i in xrange(0,len(querys)-1):
        query_token          = tokenize_stopwords_stemmer(querys[i], stemmer, True)
        terms                = search_expanded(set(query_token.split(' ')), terms_dt, matrix_tt)
        termss               = search_expanded2(set(query_token.split(' ')), terms_dt, matrix_tt) # Com expansao

        # Sem expansao
        documents_retrieval  = retrieval(terms, matrix_dt, terms_dt, query_token.split(' '))
        documents_relevants  = relevants_documents()[i+1]
        TP                   = len(documents_retrieval.intersection(documents_relevants))
        FP                   = len(documents_retrieval) - TP
        FN                   = len(documents_relevants) - TP
        TN                   = 11430 - len(documents_retrieval)
        SOMA                 = TP+FP+FN+TN
        Acuracia2            = float(len(documents_retrieval.intersection(documents_relevants)) + amount_documents - len(documents_retrieval))/float(amount_documents)

        mean_precision = mean_precision + (float(TP)/float(TP+FP))
        mean_recall    = mean_recall    + (float(TP)/float(TP+FN))
        mean_acuracy   = mean_acuracy   + (float(TP+TN)/float(TP+TN+FP+FN))
	
        # Com expansao
        documents_retrieval1 = retrieval1(termss, matrix_dt)
        documents_relevants1 = relevants_documents()[i+1]
        TP1                  = len(documents_retrieval1.intersection(documents_relevants1))
        FP1                  = len(documents_retrieval1) - TP1
        FN1                  = len(documents_relevants1) - TP1
        TN1                  = 11430 - len(documents_retrieval1)
        SOMA1 = TP1+FP1+FN1+TN1
        Acuracia21 = float(len(documents_retrieval1.intersection(documents_relevants1)) + amount_documents - len(documents_retrieval1))/float(amount_documents)

        mean_precision1 = mean_precision1 + (float(TP1)/float(TP1+FP1))
        mean_recall1    = mean_recall1    + (float(TP1)/float(TP1+FN1))
        mean_acuracy1   = mean_acuracy1   + (float(TP1+TN1)/float(TP1+TN1+FP1+FN1))

        print "Query....: " + str(i+1) + " - Sem Expansao"
        print "------------------------------------------"
        print "Documentos Recuperados: " + str(len(documents_retrieval))
        print "Documentos Relevantes : " + str(len(documents_relevants))
        print "Matriz de Confusao ( Somatorio: " + str(SOMA) + " )"
        print "TP: " + str(TP) + " | FP: " + str(FP)
        print "FN: " + str(FN) + " | TN: " + str(TN)
        print ""
        print "Precision: " + str(float(TP)/float(TP+FP))
        print "Recall...: " + str(float(TP)/float(TP+FN))
        print "Acuracia.: " + str(float(TP+TN)/float(TP+TN+FP+FN))
        print "Acuracia2: " + str(Acuracia2)
        print ""
        print "Query....: " + str(i+1) + " - Com Expansao (5)"
        print "------------------------------------------"		
        print "Documentos Recuperados: " + str(len(documents_retrieval1))
        print "Documentos Relevantes : " + str(len(documents_relevants1))
        print "Matriz de Confusao ( Somatorio: " + str(SOMA1) + " )"
        print "TP: " + str(TP1) + " | FP: " + str(FP1)
        print "FN: " + str(FN1) + " | TN: " + str(TN1)
        print ""
        print "Precision: " + str(float(TP1)/float(TP1+FP1))
        print "Recall...: " + str(float(TP1)/float(TP1+FN1))
        print "Acuracia.: " + str(float(TP1+TN1)/float(TP1+TN1+FP1+FN1))
        print "Acuracia2: " + str(Acuracia21)
        print "############################################"
        print ""
        pass
    pass
    print "****************"
    print "*** Modelo 1 ***"
    print "****************"	
    print "Precisao..: " + str(round((mean_precision/len(querys)*100),2)) + "%"
    print "Cobertura.: " + str(round((mean_recall/len(querys)*100),2)) + "%"
    print "Acuracia..: " + str(round((mean_acuracy/len(querys)*100),2)) + "%"
    print ""
    print "****************"
    print "*** Modelo 2 ***"
    print "****************"	
    print "Precisao..: " + str(round((mean_precision1/len(querys)*100),2)) + "%"
    print "Cobertura.: " + str(round((mean_recall1/len(querys)*100),2)) + "%"
    print "Acuracia..: " + str(round((mean_acuracy1/len(querys)*100),2)) + "%"

# Executar
main()

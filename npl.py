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

stemmer    = PorterStemmer()
querys     = []
querysOri  = []
expansion  = 5 # Resultado bom com 2
text_trans = []
pc1        = 0
pc2        = 1

#---------------------------------------------------------------------------------------#
def retrieval(terms, matrix_dt): # def retrieval(terms,matrix_dt, terms_dt, query): 
    result_docs = []
    for term in terms:
        sum_vector = np.sum(matrix_dt[:,term]) # Quantos documentos para cada termo
        norm = dict()
        for i in (np.where(matrix_dt[:,term]>pc1)[0]+1).tolist():   # documentos do termo
            norm[i] = float(matrix_dt[i-1, term])/float(sum_vector) # frequencia do termos no documento
        norm_sort = sorted(norm.items(), key=operator.itemgetter(1),reverse=True)
        sum_norm_sort = 0
        for i in norm_sort:
            sum_norm_sort = sum_norm_sort + i[1]
            result_docs.append(i[0])
            if sum_norm_sort >= pc2: #float(len(query))/float(len(terms)):
                break
            pass
    return set(result_docs)	

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

# Consulta expandida
def search_expanded(query, terms_dt, matrix_tt):
    terms = []
    for i in query:
        if i in terms_dt:#se botar nada a ver
            key = terms_dt.index(i) # Pega a posicao que o termo se encontra
            terms_recommended = np.sort(matrix_tt[key])[:, len(matrix_tt)-expansion:len(matrix_tt)]
            for j in terms_recommended.tolist()[0]:
                terms
                terms.append(matrix_tt[key, :].tolist()[0].index(j)) # Retorna o indice da frequencia j
            pass
            if key in terms == False or expansion == 0:
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
        key = int(line[0]) # Indice das consultas
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
    mean_precision2  = 0
    mean_recall2     = 0
    mean_acuracy2    = 0
    mean_precision3  = 0
    mean_recall3     = 0
    mean_acuracy3    = 0

    print "############################################"
    for i in xrange(0,len(querys)-1):
        query_token          = tokenize_stopwords_stemmer(querys[i], stemmer, True) # Sinonimos
        query_token2         = tokenize_stopwords_stemmer1(querys[i], stemmer)

        # retrieval
        terms                = search         (set(query_token2.split(' ')), terms_dt, matrix_tt) # Sem expansao
        terms1               = search_expanded(set(query_token2.split(' ')), terms_dt, matrix_tt) # Com expansao
        terms2               = search         (set(query_token.split(' ')), terms_dt, matrix_tt)  # Sem expansao + Sinonimos
        terms3               = search_expanded(set(query_token.split(' ')), terms_dt, matrix_tt)  # Com expansao + Sinonimos

        # Sem expansao
        documents_retrieval  = retrieval(terms, matrix_dt)  # documents_retrieval  = retrieval(terms, matrix_dt, terms_dt, query_token2.split(' '))
        documents_relevants  = relevants_documents()[i+1]
        TP                   = len(documents_retrieval.intersection(documents_relevants))
        FP                   = len(documents_retrieval) - TP
        FN                   = len(documents_relevants) - TP
        TN                   = len(matrix_dt) - len(documents_retrieval)
        SOMA                 = TP+FP+FN+TN
        Acuracia2            = float(len(documents_retrieval.intersection(documents_relevants)) + amount_documents - len(documents_retrieval))/float(amount_documents)
        doc_rel_rec          = sorted(list(documents_retrieval.intersection(documents_relevants)))

        mean_precision       = mean_precision + (float(TP)/float(TP+FP))
        mean_recall          = mean_recall    + (float(TP)/float(TP+FN))
        mean_acuracy         = mean_acuracy   + (float(TP+TN)/float(TP+TN+FP+FN))

        # Com expansao
        documents_retrieval1 = retrieval(terms1, matrix_dt)  # documents_retrieval1 = retrieval(terms1, matrix_dt, terms_dt, query_token2.split(' '))
        documents_relevants1 = relevants_documents()[i+1]
        TP1                  = len(documents_retrieval1.intersection(documents_relevants1))
        FP1                  = len(documents_retrieval1) - TP1
        FN1                  = len(documents_relevants1) - TP1
        TN1                  = len(matrix_dt) - len(documents_retrieval1)
        SOMA1                = TP1+FP1+FN1+TN1
        Acuracia21           = float(len(documents_retrieval1.intersection(documents_relevants1)) + amount_documents - len(documents_retrieval1))/float(amount_documents)
        doc_rel_rec1         = sorted(list(documents_retrieval1.intersection(documents_relevants1)))

        mean_precision1 = mean_precision1 + (float(TP1)/float(TP1+FP1))
        mean_recall1    = mean_recall1    + (float(TP1)/float(TP1+FN1))
        mean_acuracy1   = mean_acuracy1   + (float(TP1+TN1)/float(TP1+TN1+FP1+FN1))

        # Sem expansao + Sinonimos
        documents_retrieval2  = retrieval(terms2, matrix_dt)  # documents_retrieval2  = retrieval(terms2, matrix_dt, terms_dt, query_token.split(' '))
        documents_relevants2  = relevants_documents()[i+1]
        TP2                   = len(documents_retrieval2.intersection(documents_relevants2))
        FP2                   = len(documents_retrieval2) - TP2
        FN2                   = len(documents_relevants2) - TP2
        TN2                   = len(matrix_dt) - len(documents_retrieval2)
        SOMA2                 = TP2+FP2+FN2+TN2
        Acuracia22            = float(len(documents_retrieval2.intersection(documents_relevants2)) + amount_documents - len(documents_retrieval2))/float(amount_documents)
        doc_rel_rec2          = sorted(list(documents_retrieval2.intersection(documents_relevants2)))

        mean_precision2 = mean_precision2 + (float(TP2)/float(TP2+FP2))
        mean_recall2    = mean_recall2    + (float(TP2)/float(TP2+FN2))
        mean_acuracy2   = mean_acuracy2   + (float(TP2+TN2)/float(TP2+TN2+FP2+FN2))

        # Com expansao + Sinonimos
        documents_retrieval3 = retrieval(terms3, matrix_dt)  # documents_retrieval3 = retrieval(terms3, matrix_dt, terms_dt, query_token.split(' '))
        documents_relevants3 = relevants_documents()[i+1]
        TP3                  = len(documents_retrieval3.intersection(documents_relevants3))
        FP3                  = len(documents_retrieval3) - TP3
        FN3                  = len(documents_relevants3) - TP3
        TN3                  = len(matrix_dt) - len(documents_retrieval3)
        SOMA3                = TP3+FP3+FN3+TN3
        Acuracia23           = float(len(documents_retrieval3.intersection(documents_relevants3)) + amount_documents - len(documents_retrieval3))/float(amount_documents)
        doc_rel_rec3         = sorted(list(documents_retrieval3.intersection(documents_relevants3)))

        mean_precision3 = mean_precision3 + (float(TP3)/float(TP3+FP3))
        mean_recall3    = mean_recall3    + (float(TP3)/float(TP3+FN3))
        mean_acuracy3   = mean_acuracy3   + (float(TP3+TN3)/float(TP3+TN3+FP3+FN3))
        pass
    pass
    print "*******************************"
    print "Modelo 1 - Sem expansão"
    print "*******************************"
    print "Precisão..: " + str(round((mean_precision/len(querys)*100),2)) + "%"
    print "Cobertura.: " + str(round((mean_recall/len(querys)*100),2)) + "%"
    print "Acurácia..: " + str(round((mean_acuracy/len(querys)*100),2)) + "%"
    print ""
    print "*******************************"
    print "Modelo 2 - Com expansão (" + str(expansion) +")"
    print "*******************************"
    print "Precisão..: " + str(round((mean_precision1/len(querys)*100),2)) + "%"
    print "Cobertura.: " + str(round((mean_recall1/len(querys)*100),2)) + "%"
    print "Acurácia..: " + str(round((mean_acuracy1/len(querys)*100),2)) + "%"
    print ""
    print "***************************************"
    print "Modelo 3 - Sem expansão e com sinônimos"
    print "***************************************"
    print "Precisão..: " + str(round((mean_precision2/len(querys)*100),2)) + "%"
    print "Cobertura.: " + str(round((mean_recall2/len(querys)*100),2)) + "%"
    print "Acurácia..: " + str(round((mean_acuracy2/len(querys)*100),2)) + "%"
    print ""
    print "*******************************************"
    print "Modelo 4 - Com expansão e com sinônimos (" + str(expansion) +")"
    print "*******************************************"
    print "Precisão..: " + str(round((mean_precision3/len(querys)*100),2)) + "%"
    print "Cobertura.: " + str(round((mean_recall3/len(querys)*100),2)) + "%"
    print "Acurácia..: " + str(round((mean_acuracy3/len(querys)*100),2)) + "%"

############
# Executar #
############
main()

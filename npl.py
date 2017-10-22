######################################################
###### Base: NPL - National Physical Laboratory ######
######################################################
# 11429 - Documentos
# 93 - Querys
# 13 - Termos - Query 81
# 84 - Documentos - Query 41
# Termos - 7878
# https://github.com/aasouzaconsult/IA

# http://la-cci.org/

################
# Observacoes  #
################
# Verificar se todos os termos da consulta esta nos termos dos documentos (percentual)
# BM25 (Deixou mais lento) 

# Importações
import nltk
import string
import numpy as np
import pickle
import operator
import matplotlib.pyplot as pl
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import sys 
from time import time

# Declaração de variáveis globais
stemmer       = PorterStemmer()
querys        = []
querysOri     = []
text_trans    = []
text_trans1   = [] # Texto completo
termos_trans  = []
termos_trans1 = [] # Termo completo
grPrecision   = []
grPrecision1  = []
grPrecision2  = []
grPrecision3  = []
grPrecision4  = [] # BM25
grRecall      = []
grRecall1     = []
grRecall2     = []
grRecall3     = []
grRecall4     = [] # BM25

# Variáveis de parametros
expansion     = 5      # Retornam os N documentos mais relevantes por cada termo
pc1           = 0      # Parametro de configuração 1 ("Quantidade de Documentos por Termo") - Retrieval (Padrão: 0 - Todos)
pc2           = 1      # Parametro de configuração 2 ("Peso")                               - Retrieval (Padrão: 1 - Todos) | (Padrão: 0.50 - Mais relevantes)
pc3           = 100000 # Parametro de configuração 3 ("N maiores documentos retornados")    - Retrieval (Padrão: 100000 - Todos)

###########
# FUNÇÔES #
###########
#-------------------------------------------------------------------------------------------------------------#
# Função para ler o arquivo de documentos (NPL), trata o documento, tokeniza, retira StopWords e faz o Stemmer
def organizes_documents():
    files = open('npl/doc-text', 'r').read().split('/')
    for i in range(0,len(files)):
        text = files[i].strip()
        text = text.replace(str(i+1), '')
        text = text.strip()
        text_trans.append(tokenize_stopwords_stemmer(text.lower(), stemmer, False))
    generate_matrix()

#--------------------------------------------------------#
# Função para ler o arquivo de documentos originais (NPL)
def organizes_documentsOriginal():
    files = open('npl/doc-text', 'r').read().split('/')
    for i in range(0,len(files)):
        text = files[i].strip()
        text = text.replace(str(i+1), '')
        text = text.strip()
        text_trans1.append(text.lower())
    generate_matrixOriginal()

#-----------------------------------------------------------#
# Função para ler o arquivo de consultas (query) da coleção
def organizes_querys():
    files = open('npl/query-text', 'r').read().split('/')
    for i in range(0,len(files)):
        textq = files[i].strip()
        textq = textq.replace(str(i+1), '')
        textq = textq.strip()
        querys.append(textq.lower())

#--------------------------------------------------------------#
# Função para ler o arquivo de documentos relevantes da coleção
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

#----------------------------------------------------------------#
# Função para ler o arquivo de termos originais (não usado ainda)
def organizes_terms():
    files = open('npl/term-vocab', 'r').read().split('/')
    for i in range(0,len(files)):
        textt = files[i].strip()
        textt = textt.replace(str(i+1), '')
        textt = textt.strip()
        termos_trans.append(tokenize_stopwords_stemmer(textt.lower(), stemmer, False))
        termos_trans1.append(textt.lower())

#-----------------------------------------------------------------------------------------------#
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

# Processo de geração das matrizes originais - DT (Documento-Termo) e TT (Termo-Termo)
def generate_matrixOriginal(): # Tokenizado - Termos originais
    doc_term          = CountVectorizer()
    mat_doc_term      = doc_term.fit_transform(termos_trans)
    save_object(doc_term.get_feature_names(), 'termosOriginaisToken_npl.dt') # Salvando em arquivo
    # Termos originais
    doc_term1         = CountVectorizer()
    mat_doc_term1     = doc_term1.fit_transform(termos_trans1)
    save_object(doc_term1.get_feature_names(), 'termosOriginais_npl.dt')     # Salvando em arquivo

    # Texto Original
    documentOriginal_term = CountVectorizer()
    # Salvando em arquivo
    matrix_documentOriginal_term = documentOriginal_term.fit_transform(text_trans1)
    matrixOriginal_dt = np.matrix(matrix_documentOriginal_term.toarray())
    save_object(matrixOriginal_dt, 'matrixOriginal_npl.dt')
    matrixOriginal_tt = np.dot(np.transpose(matrixOriginal_dt), matrixOriginal_dt)
    save_object(matrixOriginal_tt, 'matrixOriginal_npl.tt')	
    pass

#----------------------------------------------------------------------------------------------------------------------------#
# Função de recuperação, recupera os documentos que contenham determinado termo
# A entrada da função são termos e varre a matriz termo-documento para identificar quais documentos contem o termo em questão
def retrieval(terms, matrix_dt):
    result_docs = []
    for term in terms:
        sum_vector = np.sum(matrix_dt[:,term]) # Quantos documentos para cada termo
        norm = dict()
        for i in (np.where(matrix_dt[:,term]>pc1)[0]+1).tolist():   # documentos do termo
            norm[i] = float(matrix_dt[i-1, term])/float(sum_vector) # frequencia do termos no documento
        norm_sort = sorted(norm.items(), key=operator.itemgetter(1),reverse=True)[:pc3] # os pc3 primeiros (documentos)
        sum_norm_sort = 0
        for i in norm_sort:
            sum_norm_sort = sum_norm_sort + i[1]
            result_docs.append(i[0])
            if sum_norm_sort >= pc2: # Relevancia (1 - Todos, 0.5 os 50% mais importantes)
                break
            pass
    return set(result_docs)

def retrievalteste(terms, matrix_dt): # def retrieval(terms,matrix_dt, terms_dt, query): 
    result_docs = []
    for term in terms:
        sum_vector = np.sum(matrix_dt[:,term]) # Quantos documentos para cada termo
        norm = dict()
        for i in (np.where(matrix_dt[:,term]>pc1)[0]+1).tolist():   # documentos do termo
            norm[i] = float(matrix_dt[i-1, term])/float(sum_vector) # frequencia do termos no documento
            print 'i - ' + str(i) + ' term - ' + str(term) + ' Tem ou não ' + str(float(matrix_dt[i-1, term])) + ' norm ('+ str(i) + ') -> ' + str(norm[i])			
        norm_sort  = sorted(norm.items(), key=operator.itemgetter(1),reverse=True)[:pc3] # os pc3 primeiros (documentos)
        print 'alex - ' + str(operator.itemgetter(1))
        print 'teste -> ' + str(norm_sort)
        sum_norm_sort = 0
        for i in norm_sort:
            sum_norm_sort = sum_norm_sort + i[1]
            result_docs.append(i[0])
            #print sum_norm_sort
            if sum_norm_sort >= pc2: #float(len(query))/float(len(terms)):
                break
            pass
    return set(result_docs)

#-----------------------------------------------------------------------------------------------#
# Função que tokeniza, retira stopwords (english) e faz o stemmer (Com dicionários de sinônimos)
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
        text_final +=str(stemmer.stem(text_filter[k]))
        if k != len(text_filter)-1:
            text_final+=" "
            pass
    return text_final

# Função que tokeniza, retira stopwords (english) e faz o stemmer
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
# Função de Consulta normal, pesquisa os termos da consulta na matrix co-ocorrencia
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

# Função de Consulta expandida, pesquisa os termos da consulta na matrix de co-ocorrencia
# Pega os "N expansion" mais relevantes documentos de cada termo da consulta
def search_expanded(query, terms_dt, matrix_tt):
    terms = []
    for i in query:
        if i in terms_dt:
            key = terms_dt.index(i) # Pega a posicao que o termo se encontra
            terms_recommended = np.sort(matrix_tt[key])[:, len(matrix_tt)-expansion:len(matrix_tt)] # Final da Linha os 5 ultimos colunas (maiores)
            for j in terms_recommended.tolist()[0]:
                terms
                terms.append(matrix_tt[key, :].tolist()[0].index(j)) # Retorna o indice da frequencia j
            pass
            if key in terms == False or expansion == 0:
                terms.append(key)
        pass
    pass
    return set(terms)

#----------------------------------------------------------------------------------#
# Função de Ranqueamento (BM25)
# A função precisa ser melhorada, esta com baixo desempenho e em fase de validações

# Calcula a média de documentos da coleção
def avg_documents(matrix_dt):
    soma = 0
    for i in range(0, len(matrix_dt)):
        soma += np.sum(matrix_dt[i, :])
    return float(soma)/float(len(matrix_dt))

# Função de ranqueamento
def ranqueamento(matrix_dt, terms_dt, documents_retrieval, query, avg):
        qs = QuerySearch(matrix_dt, avg)
        query_result = dict()
        for docid in documents_retrieval:
            for q in query:
                if q in terms_dt:
                    score = qs.BM25(docid,terms_dt.index(q))
                    if(not docid in query_result):
                        query_result[docid] = score
                    else:
                        query_result[docid]=query_result[docid] + score

        query_result = dict(sorted(query_result.items(), key=operator.itemgetter(1),reverse=True)).keys()[0:int(len(documents_retrieval)*0.3)]
        return query_result

# Classe de funções auxiliares para o ranqueamento
class QuerySearch(object):
    def __init__(self,matrix, avg):
        self.k = 2.0
        self.b = 0.75
        self.matrix = matrix
        self.avg = avg

    def BM25(self,docid,qid):
        df = self.calc_df(qid)
        tf = self.calc_tf(docid,qid)
        dsize = np.sum(self.matrix[docid,:])
        N = len(self.matrix)
        result = self.idf(N,df)*(tf*(self.k+1))/(tf+self.k*(1-self.b+self.b*(dsize/self.avg)))
        return result

    def idf(self,N,df):
        return np.log((N-df+0.5)/(df+0.5))

    def calc_df(self, qid):
        return len(np.where(self.matrix[:,qid]>0)[0])

    def calc_tf(self, docid,qid):
        return len(np.where(self.matrix[docid,qid]>0)[0])

#---------------------------------------------------------------------------------------#
# Função principal

def main():
    t0 = time() # Calcular o tempo de execução

    # Declaração de variáveis locais
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
    mean_precision4  = 0
    mean_recall4     = 0
    mean_acuracy4    = 0

    print "############################################"
    # Para cada uma das consultas...
    for i in xrange(0,len(querys)-1):

        # Tokeniza, retira as stopwords e faz o stemmer (opções de com ou sem sinônimos)       		
        query_token2         = tokenize_stopwords_stemmer1(querys[i], stemmer)      # Sem sinônimos
        query_token          = tokenize_stopwords_stemmer(querys[i], stemmer, True) # Com sinônimos
        query                = querys[i] # Original

        #############
        # retrieval #
        #############
        # Retorna os termos da query aplicando as funções search ou search_expanded e ainda podendo variar com ou sem sinônimos (fiz dessa forma pra comparar mais resultados em uma execução)
        terms                = search         (set(query_token2.split(' ')), terms_dt, matrix_tt)  # Sem expansao
        terms1               = search_expanded(set(query_token2.split(' ')), terms_dt, matrix_tt)  # Com expansao
        terms2               = search         (set(query_token.split(' ')) , terms_dt, matrix_tt)  # Sem expansao + Sinonimos
        terms3               = search_expanded(set(query_token.split(' ')) , terms_dt, matrix_tt)  # Com expansao + Sinonimos
        terms4               = search_expanded(set(query_token.split(' ')) , terms_dt, matrix_tt)  # Com expansao + Sinonimos (para BM25)

        # terms_orig_dt (originais usados pelo autor da base)
        #terms                = search         (set(query.split(' ')), terms_dt, matrix_tt)  # Sem expansao
        #terms1               = search_expanded(set(query.split(' ')), terms_dt, matrix_tt)  # Com expansao
        #terms2               = search         (set(query.split(' ')), terms_dt, matrix_tt)  # Sem expansao + Sinonimos
        #terms3               = search_expanded(set(query.split(' ')), terms_dt, matrix_tt)  # Com expansao + Sinonimos

        #---------------------------------------------------------------------------------------------
        # A seguir são criado 5 modelos (fiz dessa forma pra comparar mais resultados em uma execução)
        # Modelo 1 - Recupera os documentos sem utilizar expansão de termos (sem expansão)
        documents_retrieval  = retrieval(terms, matrix_dt)
        documents_relevants  = relevants_documents()[i+1]
        TP                   = len(documents_retrieval.intersection(documents_relevants)) # Verdadeiro Positivo
        FP                   = len(documents_retrieval) - TP                              # Falso Positivo
        FN                   = len(documents_relevants) - TP                              # Falso Negativo
        TN                   = len(matrix_dt) - len(documents_retrieval)                  # Verdadeiro Negativo
        SOMA                 = TP+FP+FN+TN
        Acuracia2            = float(len(documents_retrieval.intersection(documents_relevants)) + amount_documents - len(documents_retrieval))/float(amount_documents)
        doc_rel_rec          = sorted(list(documents_retrieval.intersection(documents_relevants)))

        mean_precision       = mean_precision + (float(TP)/float(TP+FP))          # Calculo da precisão da consulta
        mean_recall          = mean_recall    + (float(TP)/float(TP+FN))          # Calculo da cobertura da consulta
        mean_acuracy         = mean_acuracy   + (float(TP+TN)/float(TP+TN+FP+FN)) # Calculo da acuracia da consulta

        # Modelo 2 - Recupera os documentos utilizando expansão de termos (com expansão)
        documents_retrieval1 = retrieval(terms1, matrix_dt)
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

        # Modelo 3 - Recupera os documentos sem utilizar expansão de termos (Sem expansão e com Sinônimos)
        documents_retrieval2  = retrieval(terms2, matrix_dt)
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

        # Modelo 4 - Recupera os documentos utilizando expansão de termos (Com expansão e com Sinônimos)
        documents_retrieval3 = retrieval(terms3, matrix_dt)
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

        # Modelo 5 - Recupera os documentos utilizando expansão de termos (Com expansão, com Sinônimos e aplicando BM25)
#       documents_retrieval4   = retrieval(terms4, matrix_dt)
#       documents_retrieval4   = set(ranqueamento(matrix_dt, terms_dt, documents_retrieval4, query_token.split(' '), avg))
#       documents_relevants4   = relevants_documents()[a+1]
#       TP4                    = len(documents_retrieval4.intersection(documents_relevants4))
#       FP4                    = len(documents_retrieval4) - TP4
#       FN4                    = len(documents_relevants4) - TP4
#       TN4                    = len(matrix_dt) - len(documents_retrieval4)
#       SOMA4                  = TP4+FP4+FN4+TN4
#       Acuracia24             = float(len(documents_retrieval4.intersection(documents_relevants4)) + amount_documents - len(documents_retrieval4))/float(amount_documents)
#       doc_rel_rec4           = sorted(list(documents_retrieval4.intersection(documents_relevants4)))
#
#       mean_precision4 = mean_precision4 + (float(TP4)/float(TP4+FP4))
#       mean_recall4    = mean_recall4    + (float(TP4)/float(TP4+FN4))
#       mean_acuracy4   = mean_acuracy4   + (float(TP4+TN4)/float(TP4+TN4+FP4+FN4))

        # Gráfico (montando informações para geração dos gráficos)
        # Precisão (Precision)
        grPrecision.append(float(TP)/float(TP+FP))
        grPrecision1.append(float(TP1)/float(TP1+FP1))
        grPrecision2.append(float(TP2)/float(TP2+FP2))
        grPrecision3.append(float(TP3)/float(TP3+FP3))
       #grPrecision4.append(float(TP4)/float(TP4+FP4))

        # Cobertura (Recall)
        grRecall.append(float(TP)/float(TP+FN))
        grRecall1.append(float(TP1)/float(TP1+FN1))
        grRecall2.append(float(TP2)/float(TP2+FN2))
        grRecall3.append(float(TP3)/float(TP3+FN3))
       #grRecall4.append(float(TP4)/float(TP4+FN4))

        # Mostrando status de cada modelo na tela (por consulta - 93)
        print "Query....: " + str(i+1) + " - Sem Expansao"
        print "------------------------------------------"
        print "Qtd.Documentos Relevantes................: " + str(len(documents_relevants))
        print "Qtd.Documentos Recuperados...............: " + str(len(documents_retrieval)) + " (" + str(TP) + ")"
        print "Documentos Relevantes....................: " + str(documents_relevants)
        print "Documentos Recuperados e Relevantes......: " + str(doc_rel_rec)
        print "Documentos Relevantes e NÃO Recuperados..: " + str(sorted(list(set(documents_relevants).difference(set(doc_rel_rec)))))
        print "Matriz de Confusao ( Somatorio: " + str(SOMA) + " )"
        print "TP: " + str(TP) + " | FP: " + str(FP)
        print "FN: " + str(FN) + " | TN: " + str(TN)
        print ""
        print "Precision: " + str(float(TP)/float(TP+FP))
        print "Recall...: " + str(float(TP)/float(TP+FN))
        print "Acuracia.: " + str(float(TP+TN)/float(TP+TN+FP+FN))
        print "----------------------------------------------"
        print "Query....: " + str(i+1) + " - Com Expansao (" + str(expansion) +")"
        print "----------------------------------------------"		
        print "Qtd.Documentos Relevantes................: " + str(len(documents_relevants1))
        print "Qtd.Documentos Recuperados...............: " + str(len(documents_retrieval1)) + " (" + str(TP1) + ")"
        print "Documentos Relevantes....................: " + str(documents_relevants1)
        print "Documentos Recuperados e Relevantes......: " + str(doc_rel_rec1)
        print "Documentos Relevantes e NÃO Recuperados..: " + str(sorted(list(set(documents_relevants1).difference(set(doc_rel_rec1)))))
        print "Matriz de Confusao ( Somatorio: " + str(SOMA1) + " )"
        print "TP: " + str(TP1) + " | FP: " + str(FP1)
        print "FN: " + str(FN1) + " | TN: " + str(TN1)
        print ""
        print "Precision: " + str(float(TP1)/float(TP1+FP1))
        print "Recall...: " + str(float(TP1)/float(TP1+FN1))
        print "Acuracia.: " + str(float(TP1+TN1)/float(TP1+TN1+FP1+FN1))
        print "------------------------------------------------------"
        print "Query....: " + str(i+1) + " - Sem Expansao + Sinonimos"
        print "------------------------------------------------------"
        print "Qtd.Documentos Relevantes................: " + str(len(documents_relevants2))
        print "Qtd.Documentos Recuperados...............: " + str(len(documents_retrieval2)) + " (" + str(TP2) + ")"
        print "Documentos Relevantes....................: " + str(documents_relevants2)
        print "Documentos Recuperados e Relevantes......: " + str(doc_rel_rec2)
        print "Documentos Relevantes e NÃO Recuperados..: " + str(sorted(list(set(documents_relevants2).difference(set(doc_rel_rec2)))))
        print "Matriz de Confusao ( Somatorio: " + str(SOMA2) + " )"
        print "TP: " + str(TP2) + " | FP: " + str(FP2)
        print "FN: " + str(FN2) + " | TN: " + str(TN2)
        print ""
        print "Precision: " + str(float(TP2)/float(TP2+FP2))
        print "Recall...: " + str(float(TP2)/float(TP2+FN2))
        print "Acuracia.: " + str(float(TP2+TN2)/float(TP2+TN2+FP2+FN2))
        print "----------------------------------------------------------"
        print "Query....: " + str(i+1) + " - Com Expansao + Sinonimos (" + str(expansion) +")"
        print "----------------------------------------------------------"
        print "Qtd.Documentos Relevantes................: " + str(len(documents_relevants3))
        print "Qtd.Documentos Recuperados...............: " + str(len(documents_retrieval3)) + " (" + str(TP3) + ")"
        print "Documentos Relevantes....................: " + str(documents_relevants3)
        print "Documentos Recuperados e Relevantes......: " + str(doc_rel_rec3)
        print "Documentos Relevantes e NÃO Recuperados..: " + str(sorted(list(set(documents_relevants3).difference(set(doc_rel_rec3)))))
        print "Matriz de Confusao ( Somatorio: " + str(SOMA3) + " )"
        print "TP: " + str(TP3) + " | FP: " + str(FP3)
        print "FN: " + str(FN3) + " | TN: " + str(TN3)
        print ""
        print "Precision: " + str(float(TP3)/float(TP3+FP3))
        print "Recall...: " + str(float(TP3)/float(TP3+FN3))
        print "Acuracia.: " + str(float(TP3+TN3)/float(TP3+TN3+FP3+FN3))
#       print "------------------------------------------------------------------"
#       print "Query....: " + str(i+1) + " - Com Expansao + Sinonimos usando BM25"
#       print "------------------------------------------------------------------"
#       print "Qtd.Documentos Relevantes................: " + str(len(documents_relevants4))
#       print "Qtd.Documentos Recuperados...............: " + str(len(documents_retrieval4)) + " (" + str(TP4) + ")"
#       print "Documentos Relevantes....................: " + str(documents_relevants4)
#       print "Documentos Recuperados e Relevantes......: " + str(doc_rel_rec4)
#       print "Documentos Relevantes e NÃO Recuperados..: " + str(sorted(list(set(documents_relevants4).difference(set(doc_rel_rec4)))))
#       print "Matriz de Confusao ( Somatorio: " + str(SOMA4) + " )"
#       print "TP: " + str(TP4) + " | FP: " + str(FP4)
#       print "FN: " + str(FN4) + " | TN: " + str(TN4)
#       print ""
#       print "Precision: " + str(float(TP4)/float(TP4+FP4))
#       print "Recall...: " + str(float(TP4)/float(TP4+FN4))
#       print "Acuracia.: " + str(float(TP4+TN4)/float(TP4+TN4+FP4+FN4))
        print "############################################"
        print ""
        pass
    pass

    # Mostrando status de cada modelo na tela (por consulta - 93)
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
    print ""    
    print "*******************************************"    
#   print "Modelo 5 - Com expansão, com sinônimos e usando BM25 (" + str(expansion) +")"
#   print "*******************************************"
#   print "Precisão..: " + str(round((mean_precision4/len(querys)*100),2)) + "%"
#   print "Cobertura.: " + str(round((mean_recall4/len(querys)*100),2)) + "%"
#   print "Acurácia..: " + str(round((mean_acuracy4/len(querys)*100),2)) + "%"
#   print "*******************************************"
    print("done in %fs" % (time() - t0)) # Mostra o tempo de execução do main()

############
# Executar #
############
# Apenas a primeira vez para gerar os objetos (demorada: +- 8hs)
#organizes_documents()         # Carregar o arquivo de documentos e objeto das matrizes

# Original (teste)
#organizes_terms()             # Originais
#organizes_documentsOriginal() # Originais

#matrix_dt           = load_object('objects/matrixOriginal_npl.dt')
#matrix_tt           = load_object('objects/matrixOriginal_npl.tt')
#terms_dt            = load_object('objects/termosOriginais_npl.dt')  #load_object('objects/termosOriginaisToken_npl.dt')

# Execução normal
# ---------------
#organizes_querys() # Carregar o arquivo de consulta (Query)

# Carregar as matrizes em variáveis
#matrix_dt           = load_object('objects/matrix_npl.dt')
#matrix_tt           = load_object('objects/matrix_npl.tt')
#terms_dt            = load_object('objects/terms_npl.dt')

# Media de documentos para ser utilizado pelo BM25
#avg = avg_documents(matrix_dt)

# Execução (mostra resultados e etc)
main()

# Geração do gráfico (Será aberto em uma nova janela)
pl.clf()
pl.plot(np.sort(grRecall), np.sort(grPrecision)  , label='Sem expansao')
pl.plot(np.sort(grRecall1), np.sort(grPrecision1), label='Com expansao')
pl.plot(np.sort(grRecall2), np.sort(grPrecision2), label='Sem expansao + Sinonimos')
pl.plot(np.sort(grRecall3), np.sort(grPrecision3), label='Com expansao + Sinonimos')
#pl.plot(np.sort(grRecall4), np.sort(grPrecision4), label='Com expansao + Sinonimos + BM25')
pl.xlabel('Recall')
pl.ylabel('Precision')
#pl.ylim([0.0, 1.0])
#pl.xlim([0.0, 1.0])
pl.title('Precision-Recall - Colecao: NPL')
pl.legend(loc="upper left")
pl.show()


##########
# Testes #
##########

teste(0)

def teste(a):
   # t1 - Teste 1
   query_token_1    = tokenize_stopwords_stemmer1(querys[a], stemmer)
   term_token_1     = search(query_token_1.split(' '), terms_dt, matrix_tt)
   
   documents_retrievalt1  = retrieval(term_token_1, matrix_dt)
   documents_relevantst1  = relevants_documents()[a+1]
   
   TPt1         = len(documents_retrievalt1.intersection(documents_relevantst1))
   FPt1         = len(documents_retrievalt1) - TPt1
   FNt1         = len(documents_relevantst1) - TPt1
   TNt1         = 11430 - len(documents_retrievalt1)
   SOMAt1       = TPt1+FPt1+FNt1+TNt1
   documents_relevants_recuperados1 = sorted(list(documents_retrievalt1.intersection(documents_relevantst1)))

   print "* Teste 1 *"
   print "Query....: " + str(a+1) + " * Sem Expansao *"
   print "Qtd.Documentos Relevantes................: " + str(len(documents_relevantst1))
   print "Qtd.Documentos Recuperados...............: " + str(len(documents_retrievalt1)) + " (" + str(TPt1) + ")"
   print "Documentos Relevantes....................: " + str(documents_relevantst1)
   print "Documentos Recuperados e Relevantes......: " + str(documents_relevants_recuperados1)
   print "Documentos Relevantes e NÃO Recuperados..: " + str(sorted(list(set(documents_relevantst1).difference(set(documents_relevants_recuperados1)))))
   print "Matriz de Confusao ( Somatorio: " + str(SOMAt1) + " )"
   print "TP: " + str(TPt1) + " | FP: " + str(FPt1)
   print "FN: " + str(FNt1) + " | TN: " + str(TNt1)
   print "Precision: " + str(float(TPt1)/float(TPt1+FPt1))
   print "Recall...: " + str(float(TPt1)/float(TPt1+FNt1))
   print "Acuracia.: " + str(float(TPt1+TNt1)/float(TPt1+TNt1+FPt1+FNt1))
   print "############################################"
   
   # t2 - Teste 2
   query_token_2    = tokenize_stopwords_stemmer1(querys[a], stemmer)
   term_token_2     = search_expanded(set(query_token_2.split(' ')), terms_dt, matrix_tt)
   
   documents_retrievalt2  = retrieval(term_token_2, matrix_dt)  # documents_retrievalt2  = retrieval(term_token_2, matrix_dt, terms_dt, query_token_2.split(' '))
   documents_relevantst2  = relevants_documents()[a+1]
   
   TPt2         = len(documents_retrievalt2.intersection(documents_relevantst2))
   FPt2         = len(documents_retrievalt2) - TPt2
   FNt2         = len(documents_relevantst2) - TPt2
   TNt2         = 11430 - len(documents_retrievalt2)
   SOMAt2       = TPt2+FPt2+FNt2+TNt2
   documents_relevants_recuperados2 = sorted(list(documents_retrievalt2.intersection(documents_relevantst2)))
   		
   print "* Teste 2 *"
   print "Query....: " + str(a+1) + " * Sem Expansao *"
   print "Qtd.Documentos Relevantes................: " + str(len(documents_relevantst2))
   print "Qtd.Documentos Recuperados...............: " + str(len(documents_retrievalt2)) + " (" + str(TPt2) + ")"
   print "Documentos Relevantes....................: " + str(documents_relevantst2)
   print "Documentos Recuperados e Relevantes......: " + str(documents_relevants_recuperados2)
   print "Documentos Relevantes e NÃO Recuperados..: " + str(sorted(list(set(documents_relevantst2).difference(set(documents_relevants_recuperados2)))))
   print "Matriz de Confusao ( Somatorio: " + str(SOMAt2) + " )"
   print "TP: " + str(TPt2) + " | FP: " + str(FPt2)
   print "FN: " + str(FNt2) + " | TN: " + str(TNt2)
   print "Precision: " + str(float(TPt2)/float(TPt2+FPt2))
   print "Recall...: " + str(float(TPt2)/float(TPt2+FNt2))
   print "Acuracia.: " + str(float(TPt2+TNt2)/float(TPt2+TNt2+FPt2+FNt2))
   print "############################################"

   # t3 - Teste 3
   query_token_3    = tokenize_stopwords_stemmer(querys[a], stemmer, True)
   term_token_3     = search_expanded(set(query_token_3.split(' ')), terms_dt, matrix_tt)

   documents_retrievalt3  = retrieval(term_token_3, matrix_dt)
   documents_relevantst3  = relevants_documents()[a+1]

   TPt3         = len(documents_retrievalt3.intersection(documents_relevantst3))
   FPt3         = len(documents_retrievalt3) - TPt3
   FNt3         = len(documents_relevantst3) - TPt3
   TNt3         = 11430 - len(documents_retrievalt3)
   SOMAt3       = TPt3+FPt3+FNt3+TNt3
   documents_relevants_recuperados3 = sorted(list(documents_retrievalt3.intersection(documents_relevantst3)))
   
   print "* Teste 3 *"
   print "Query....: " + str(a+1) + " * Sem Expansao *"
   print "Qtd.Documentos Relevantes................: " + str(len(documents_relevantst3))
   print "Qtd.Documentos Recuperados...............: " + str(len(documents_retrievalt3)) + " (" + str(TPt3) + ")"
   print "Documentos Relevantes....................: " + str(documents_relevantst3)
   print "Documentos Recuperados e Relevantes......: " + str(documents_relevants_recuperados3)
   print "Documentos Relevantes e NÃO Recuperados..: " + str(sorted(list(set(documents_relevantst3).difference(set(documents_relevants_recuperados3)))))
   print "Matriz de Confusao ( Somatorio: " + str(SOMAt3) + " )"
   print "TP: " + str(TPt3) + " | FP: " + str(FPt3)
   print "FN: " + str(FNt3) + " | TN: " + str(TNt3)
   print "Precision: " + str(float(TPt3)/float(TPt3+FPt3))
   print "Recall...: " + str(float(TPt3)/float(TPt3+FNt3))
   print "Acuracia.: " + str(float(TPt3+TNt3)/float(TPt3+TNt3+FPt3+FNt3))
   print "############################################"

   # t4 - Teste 4
   query_token_4    = tokenize_stopwords_stemmer(querys[a], stemmer, True)
   term_token_4     = search(set(query_token_4.split(' ')), terms_dt, matrix_tt)

   documents_retrievalt4  = retrieval(term_token_4, matrix_dt) #documents_retrievalt4  = retrievalteste(term_token_4, matrix_dt)
   documents_relevantst4  = relevants_documents()[a+1]

   TPt4         = len(documents_retrievalt4.intersection(documents_relevantst4))
   FPt4         = len(documents_retrievalt4) - TPt4
   FNt4         = len(documents_relevantst4) - TPt4
   TNt4         = 11430 - len(documents_retrievalt4)
   SOMAt4       = TPt4+FPt4+FNt4+TNt4
   documents_relevants_recuperados4 = sorted(list(documents_retrievalt4.intersection(documents_relevantst4)))

   print "* Teste 4 *"
   print "Query....: " + str(a+1) + " * Com Expansao (5) *"
   print "Qtd.Documentos Relevantes................: " + str(len(documents_relevantst4))
   print "Qtd.Documentos Recuperados...............: " + str(len(documents_retrievalt4)) + " (" + str(TPt4) + ")"
   print "Documentos Relevantes....................: " + str(documents_relevantst4)
   print "Documentos Recuperados e Relevantes......: " + str(documents_relevants_recuperados4)
   print "Documentos Relevantes e NÃO Recuperados..: " + str(sorted(list(set(documents_relevantst4).difference(set(documents_relevants_recuperados4)))))
   print "Matriz de Confusao ( Somatorio: " + str(SOMAt4) + " )"
   print "TP: " + str(TPt4) + " | FP: " + str(FPt4)
   print "FN: " + str(FNt4) + " | TN: " + str(TNt4)
   print "Precision: " + str(float(TPt4)/float(TPt4+FPt4))
   print "Recall...: " + str(float(TPt4)/float(TPt4+FNt4))
   print "Acuracia.: " + str(float(TPt4+TNt4)/float(TPt4+TNt4+FPt4+FNt4))
   print "############################################"
   
   # t5 - Teste 5
   query_token  = tokenize_stopwords_stemmer(querys[a], stemmer, True) # Sinonimos
   terms4       = search_expanded(set(query_token.split(' ')) , terms_dt, matrix_tt)  # Com expansao + Sinonimos
  
   # Com expansao + Sinonimos
   documents_retrieval4   = retrieval(terms4, matrix_dt)
   documents_retrieval4   = set(ranqueamento(matrix_dt, terms_dt, documents_retrieval4, query_token.split(' '), avg))
   documents_relevants4   = relevants_documents()[a+1]
   TP4                    = len(documents_retrieval4.intersection(documents_relevants4))
   FP4                    = len(documents_retrieval4) - TP4
   FN4                    = len(documents_relevants4) - TP4
   TN4                    = len(matrix_dt) - len(documents_retrieval4)
   SOMA4                  = TP4+FP4+FN4+TN4
   Acuracia24             = float(len(documents_retrieval4.intersection(documents_relevants4)) + amount_documents - len(documents_retrieval4))/float(amount_documents)
   doc_rel_rec4           = sorted(list(documents_retrieval4.intersection(documents_relevants4)))

   mean_precision4 = mean_precision4 + (float(TP4)/float(TP4+FP4))
   mean_recall4    = mean_recall4    + (float(TP4)/float(TP4+FN4))
   mean_acuracy4   = mean_acuracy4   + (float(TP4+TN4)/float(TP4+TN4+FP4+FN4))

   print "* Teste 5 *"
   print "Query....: " + str(a+1) + " * Com Expansao (5) *"
   print "Qtd.Documentos Relevantes................: " + str(len(documents_relevants4))
   print "Qtd.Documentos Recuperados...............: " + str(len(documents_retrieval4)) + " (" + str(TP4) + ")"
   print "Documentos Relevantes....................: " + str(documents_relevants4)
   print "Documentos Recuperados e Relevantes......: " + str(doc_rel_rec4)
   print "Documentos Relevantes e NÃO Recuperados..: " + str(sorted(list(set(documents_relevants4).difference(set(doc_rel_rec4)))))
   print "Matriz de Confusao ( Somatorio: " + str(SOMA4) + " )"
   print "TP: " + str(TP4) + " | FP: " + str(FP4)
   print "FN: " + str(FN4) + " | TN: " + str(TN4)
   print "Precision: " + str(float(TP4)/float(TP4+FP4))
   print "Recall...: " + str(float(TP4)/float(TP4+FN4))
   print "Acuracia.: " + str(float(TP4+TN4)/float(TP4+TN4+FP4+FN4))
   print "############################################"

############
# ANALISES #
############
arquivo = open('teste.txt', 'w')
arquivo.writelines(text_trans)
arquivo.close()

retrievalteste(term_token_4, matrix_dt)
print 'valor - ' + str(np.where(matrix_dt[:,5278]>0))

# Achar termo
terms_dt[5278]


##########
# Extras #
##########

# Termos em querys sem conter no Texto
document_term = CountVectorizer()
matrix_querys_term = document_term.fit_transform(querys)
save_object(document_term.get_feature_names(), 'Qterms_npl.dt')
Qterms_dt = load_object('objects/Qterms_npl.dt') # termos de pesquisa - 346 (termos geral: 7878)

TermosQSemRef = list(set(Qterms_dt) - set(terms_dt))
# 3 [u'transistoris', u'optimis', u'pretreat']
	

########################
# Similaridade Cosseno #
########################

# Documentos
document_term = CountVectorizer()
TfIdf   = document_term.fit_transform(text_trans) # Documento X Termo]
M_TfIdf = TfIdf.transpose() # Termo X Documento

# Querys
query_trans = []
#query_trans.append(querys[2])
query_trans.append(tokenize_stopwords_stemmer(querys[2], stemmer))

# Tf-Idf da Query
Q_TfIdf  = document_term.fit_transform(query_trans)

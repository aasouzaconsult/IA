import numpy as np
import pickle
import operator
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from nltk import word_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
k = 3

#def preprocess(text):
#    text = text.lower()
#    doc = word_tokenize(text)
#    doc = [word for word in doc if word not in stop_words]
#    doc = [word for word in doc if word.isalpha()]
#    return doc

ng20 = fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes'))
texts = ng20.data
count = 1000  #(max = 18846)

document_term = TfidfVectorizer()
matrix_document_term = document_term.fit_transform(texts[0:count]).toarray()
matrix_document_document = np.dot(matrix_document_term, np.transpose(matrix_document_term))
matrix_adj = np.zeros(shape=matrix_document_document.shape)

for i in range(len(matrix_document_document)):
    od = sorted(dict(enumerate(matrix_document_document[i])).items(),key=operator.itemgetter(1),reverse=True)[0:k]
    for j in od:
        matrix_adj[i,j[0]] = 1

# with open('matrix_document_document', 'wb') as output:
#     pickle.dump(matrix_document_document, output, pickle.HIGHEST_PROTOCOL)

G = nx.from_numpy_matrix(matrix_adj)
#nx.draw(G)
#plt.show()
nx.draw(G, width=1, font_size=16, with_labels=True, alpha=0.4, node_color=range(1000))
plt.show()

# Agrupamentos
nx.clustering(G)

# Isomorfismo
G5 = nx.complete_graph(5)
G10 = nx.complete_graph(10)
nx.draw(G5)
nx.draw(G10, node_color='b')

nx.is_isomorphic(G5,G10)

# Tentativa de encontrar Isomorfismo
for i in range(10):
	for j in range (10):
		print(i, j, nx.is_isomorphic(nx.complete_graph(i), nx.complete_graph(j)))

###########
# Estudos #
###########

# Grau
nx.degree(G)

# Verificar um vértice e suas relações
a = nx.complete_graph(5)
nx.draw(a)

# Bipartido
K_3_5=nx.complete_bipartite_graph(3,5)
nx.draw(K_3_5)

# Outros
G.number_of_nodes()
G.number_of_edges()

G.nodes()
G.edges()
G.neighbors(1)

nx.draw_spectral(a)
nx.draw_random(a)
nx.draw_circular(a)

# Referências
# https://networkx.github.io/documentation/networkx-1.10/tutorial/tutorial.html
# https://www.csie.ntu.edu.tw/~azarc/sna/networkx/networkx/algorithms/

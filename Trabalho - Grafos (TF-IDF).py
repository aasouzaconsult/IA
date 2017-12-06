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

# Selecionar apenas as categorias (http://scikit-learn.org/stable/datasets/twenty_newsgroups.html)
categories = ['sci.electronics', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']

ng20 = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes')) # subset='all'

#ver as categorias
ng20.target_names
# ou (individual - Exemplo, 10 primeiros)
for t in twenty_train.target[:10]:
   print(twenty_train.target_names[t])

texts = ng20.data
# count = 1000  #(max = 18846) # Pegando agora 3 categorias (1759 Docs)

document_term = TfidfVectorizer()
# matrix_document_term = document_term.fit_transform(texts[0:count]).toarray()
matrix_document_term = document_term.fit_transform(texts).toarray()
matrix_document_document = np.dot(matrix_document_term, np.transpose(matrix_document_term))
matrix_adj = np.zeros(shape=matrix_document_document.shape)

for i in range(len(matrix_document_document)):
    od = sorted(dict(enumerate(matrix_document_document[i])).items(),key=operator.itemgetter(1),reverse=True)[0:k]
    for j in od:
        matrix_adj[i,j[0]] = 1

# with open('matrix_document_document', 'wb') as output:
#     pickle.dump(matrix_document_document, output, pickle.HIGHEST_PROTOCOL)

G = nx.from_numpy_matrix(matrix_adj)
nx.draw(G, width=1, font_size=16, with_labels=True, alpha=0.4, node_color=range(1000))
plt.show()

# Test (Completo)
texts[7268]
texts[2523]
		
texts[3793]
texts[5655]

# Agrupamentos
nx.clustering(G)

from sklearn.cluster import SpectralClustering
sc = SpectralClustering(20, affinity='precomputed', n_init=100)
sc.fit(matrix_adj)
print(sc.labels_)
np.savetxt('text.txt',mat,fmt='%.2f')

# Isomorfismo
G5 = nx.complete_graph(5)
G10 = nx.complete_graph(10)
nx.draw(G5, width=1, font_size=16, with_labels=True, alpha=0.4)
nx.draw(G10, width=1, font_size=16, with_labels=True, alpha=0.4, node_color='b')

nx.is_isomorphic(G10,G10)

# Tentativa de encontrar Isomorfismo
for i in range(10):
	for j in range (10):
		print(i, j, nx.is_isomorphic(nx.complete_graph(i), nx.complete_graph(j)))

###########
# Estudos #
###########

# Caminho mínimo
nx.average_shortest_path_length(G)

# Grau
nx.degree(G)

# Verificar um vértice e suas relações
a = nx.complete_graph(5)
nx.draw(a, width=1, font_size=16, with_labels=True, alpha=0.4)

# Bipartido
K_3_5=nx.complete_bipartite_graph(3,5)
nx.draw(K_3_5, width=1, font_size=16, with_labels=True, alpha=0.4)
#or
import networkx as nx
from networkx.algorithms import bipartite
B = nx.Graph()
B.add_nodes_from([4,5,7], bipartite=0) # Add the node attribute "bipartite"
B.add_nodes_from([654, 611], bipartite=1)
B.add_edges_from([(4,611), (4,4), (4,654), (5,5), (5,654), (7,7), (7,654),(654,654),(611,611)])
nx.draw_circular(B, width=1, font_size=16, with_labels=True, alpha=0.4, node_color=range(5))

# Subgrafo (testar)
res = [4,5,7,654,611]
pos = nx.spring_layout(G)
k = G.subgraph(res)
nx.draw_networkx(k, pos=pos, node_color='b')
othersubgraph = G.subgraph(range(4,G.order()))
nx.draw_networkx(othersubgraph, pos=pos, node_color = 'r')

# Auxiliares
nx.info(G)
nx.density(G)

G.number_of_nodes()
G.number_of_edges()

G.nodes()
G.edges()
G.neighbors(1)

##########
# Outros #
##########

# Digrafo
H = nx.DiGraph(G)
list(H.edges())
edgelist = [(0, 1), (1, 2), (2, 3)]
H = nx.Graph(edgelist)
nx.draw(H)

# Referências
# https://networkx.github.io/documentation/networkx-1.10/tutorial/tutorial.html
# https://networkx.github.io/documentation/networkx-1.10/reference/functions.html
# https://networkx.github.io/documentation/latest/tutorial.html
# https://networkx.github.io/documentation/networkx-1.10/reference/algorithms.shortest_paths.html
# https://www.csie.ntu.edu.tw/~azarc/sna/networkx/networkx/algorithms/ *
# https://stackoverflow.com/questions/24829123/plot-bipartite-graph-using-networkx-in-python
# http://scikit-learn.org/stable/tutorial/statistical_inference/unsupervised_learning.html

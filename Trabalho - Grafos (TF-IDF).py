import numpy as np
import pickle
import operator
import random
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from nltk import word_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
count      = 2929 # Pegando agora 3 categorias (2929 Docs) / maximo = 18846
k          = 3    # Pegar as 3 maiores relações
k_centers  = 2 

# Selecionar apenas as categorias (http://scikit-learn.org/stable/datasets/twenty_newsgroups.html)
categories = ['sci.electronics', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']

ng20 = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes')) # subset='all'
ng20.target_names

texts = ng20.data

# Vazios ou em falha - Coloca uma frase padrão
for i in range(len(texts)):
    if len(texts[i]) <= 20:
        texts[i] = "no text - empty document"

# Matrix - TF-IDF
document_term = TfidfVectorizer()
matrix_document_term = document_term.fit_transform(texts).toarray()
matrix_document_document = np.dot(matrix_document_term, np.transpose(matrix_document_term))

# Matrix de Adjacências
matrix_adj = np.zeros(shape=matrix_document_document.shape) # Matriz de zeros no tamanho da matrix DxD

for i in range(len(matrix_document_document)):
    od = sorted(dict(enumerate(matrix_document_document[i])).items(),key=operator.itemgetter(1),reverse=True)[1:k]
    for j in od:
        matrix_adj[i,j[0]] = j[1]

# Salvando...
# with open('matrix_document_document', 'wb') as output:
#     pickle.dump(matrix_document_document, output, pickle.HIGHEST_PROTOCOL)

# Grafo
G = nx.from_numpy_matrix(matrix_adj)
nx.draw(G, width=1, font_size=16, with_labels=True, alpha=0.4, node_color=range(count))
plt.show()

# Clustering
centroids = random.sample(range(count), k_centers)

#################
# ? Pesquisas ? #
#################
# - Colorir grupos
# - Grafos separados (o que é?)
# - K Dijkstra

######################
# Análises e Estudos #
######################
# Informações do Grafo
nx.info(G)
nx.density(G)

G.number_of_nodes()
G.number_of_edges()

G.nodes()
G.edges()
G.neighbors(1)

# Grau
nx.degree(G)

# Verificar um vértice e suas relações
a = nx.complete_graph(5)
nx.draw(a, width=1, font_size=16, with_labels=True, alpha=0.4)

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

# Caminho mínimo
nx.average_shortest_path_length(G)

# de 0 a 4
nx.shortest_path(G,source=0,target=4)
print(nx.shortest_path_length(G,source=0,target=4))

nx.dijkstra_path(G,source=0,target=4)
nx.single_source_dijkstra(G, source=0) # do zero para todos os vértices

nx.bellman_ford(G, source=0)
nx.floyd_warshall(G) # lento
nx.astar_path(G,source=0,target=4)

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

##########
# Outros #
##########
# Digrafo
H = nx.DiGraph(G)
list(H.edges())
edgelist = [(0, 1), (1, 2), (2, 3)]
H = nx.Graph(edgelist)
nx.draw(H)

# Teste - Matriz de Adjacências
adj_matrix = np.matrix([[0,1,0,0,1,0],[1,0,1,0,1,0],[0,1,0,1,0,0],[0,0,1,0,1,1],[1,1,0,1,0,0],[0,0,0,1,0,0]])
H = nx.from_numpy_matrix(adj_matrix)
nx.draw(H, width=1, font_size=16, with_labels=True, alpha=0.4)

# ver as categorias
# ou (individual - Exemplo, 10 primeiros)
# for t in ng20.target[:10]:
#    print(ng20.target_names[t])

#def preprocess(text):
#    text = text.lower()
#    doc = word_tokenize(text)
#    doc = [word for word in doc if word not in stop_words]
#    doc = [word for word in doc if word.isalpha()]
#    return doc

# Referências
# https://networkx.github.io/documentation/networkx-1.10/tutorial/tutorial.html
# https://networkx.github.io/documentation/networkx-1.10/reference/functions.html
# https://networkx.github.io/documentation/latest/tutorial.html
# https://networkx.github.io/documentation/networkx-1.10/reference/algorithms.shortest_paths.html
# https://www.csie.ntu.edu.tw/~azarc/sna/networkx/networkx/algorithms/ *
# https://stackoverflow.com/questions/24829123/plot-bipartite-graph-using-networkx-in-python
# http://scikit-learn.org/stable/tutorial/statistical_inference/unsupervised_learning.html

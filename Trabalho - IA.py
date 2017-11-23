import re
from unicodedata import normalize

def remover_acentos(txt, codif='utf-8'):
    return normalize('NFKD', txt.decode(codif)).encode('ASCII', 'ignore')

s = "A clusterização de documentos é um problema que consiste em encontrar grupos de documentos, dado um coleção, com características semelhantes. No trabalho, usaremos uma coleção de documentos (20newsgroups), onde cada documento dessa coleção será representado por um grafo de dependência. Será aplicado técnicas de similaridade (como similaridade cosseno, distância ou k-means) para identificar os grafos (documentos) com características semelhantes."

t = remover_acentos(s)

# Remove os espaços em branco
t = re.sub(r'\s', '', t)

# Voltar os espaços

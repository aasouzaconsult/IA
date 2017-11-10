# http://www.nltk.org/book/ch05.html
# http://www.nltk.org/book/ch07.html
# http://scikit-learn.org/stable/modules/biclustering.html

locs = [( 'Omnicom' , 'IN' , 'New York' ),
        ( 'DDB Needham' , 'IN' , 'New York' ),
        ( 'Kaplan Thaler Group' , 'IN' , 'New York' ),
        ( 'BBDO Sul' , 'IN' , 'Atlanta' ),
        ( 'Georgia-Pacific' , 'IN' , 'Atlanta' )]
query = [e1 for (e1, rel, e2) in locs if e2=='Atlanta']
print (query)
##################################################################

import nltk
groucho_dep_grammar = nltk.DependencyGrammar.fromstring("""
   'shot' -> 'I' | 'elephant' | 'in'
   'elephant' -> 'an' | 'in'
   'in' -> 'pajamas'
   'pajamas' -> 'my'
""")
print(groucho_dep_grammar)
pdp = nltk.ProjectiveDependencyParser(groucho_dep_grammar)
sent = 'I shot an elephant in my pajamas'.split()
trees = pdp.parse(sent)
for tree in trees:
   print(tree)

################################################################## 
# Chunking (grafo)
# the little yellow dog barked at the cat
# o pequeno cão amarelo latiu no gato

sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),
("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")] 

grammar = "NP: {<DT>?<JJ>*<NN>}" # Esta regra diz que um pedaço NP deve ser formado quando o 
#chunker encontra um determinador opcional (optional determiner) ( DT ), seguido por qualquer número de adjetivos ( JJ) 
#e, em seguida, um substantivo ( NN ).

cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
result.draw()

##################################################################
# Mais completo

import nltk
from nltk import word_tokenize
text = word_tokenize("the little yellow dog barked at the cat")
text = word_tokenize("I wonder how many atheists out there care to speculateon the face of the world.")
sentence = nltk.pos_tag(text)

grammar = "NP: {<DT>?<JJ>*<NN>}" # dog e cat

# http://blog.quibb.org/2010/01/nltk-regular-expression-parser-regexpparser/
grammar = """
	NP:   {<PRP>?<JJ.*>*<NN.*>+}
	CP:   {<JJR|JJS>}
	VERB: {<VB.*>}
	THAN: {<IN>}
	COMP: {<DT>?<NP><RB>?<VERB><DT>?<CP><THAN><DT>?<NP>}
	"""

cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
result.draw()

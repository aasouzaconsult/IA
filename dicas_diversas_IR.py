#########
# Livro #
#########
# http://www.ling.helsinki.fi/kit/2008s/clt231/nltk-0.9.5/doc/en/book.html

#####################
# Janela deslizante #
#####################
sent = ['The', 'dog', 'gave', 'John', 'the', 'newspaper']
n = 3
[sent[i:i+n] for i in range(len(sent)-n+1)]

##########
# N-Gram #
##########
# http://www.ling.helsinki.fi/kit/2008s/clt231/nltk-0.9.5/doc/en/book.html#n_gram_tagger_index_term
# http://tetration.xyz/Ngram-Tutorial/

# https://programminghistorian.org/lessons/keywords-in-context-using-n-grams
import obo

wordstring = 'it was the best of times it was the worst of times '
wordstring += 'it was the age of wisdom it was the age of foolishness'

allMyWords = wordstring.split()
print(obo.getNGrams(allMyWords, 3))

#####################################################
# Output Data as an HTML File with Python (wrapper) #
#####################################################
# https://programminghistorian.org/lessons/output-data-as-html-file

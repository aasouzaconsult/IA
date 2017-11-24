# Livro
# http://www.ling.helsinki.fi/kit/2008s/clt231/nltk-0.9.5/doc/en/book.html

#N-Gram
#http://www.ling.helsinki.fi/kit/2008s/clt231/nltk-0.9.5/doc/en/book.html#n_gram_tagger_index_term
#http://tetration.xyz/Ngram-Tutorial/

# Janela deslizante
sent = ['The', 'dog', 'gave', 'John', 'the', 'newspaper']
n = 3
[sent[i:i+n] for i in range(len(sent)-n+1)]

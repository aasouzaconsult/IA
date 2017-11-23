# http://2017.compciv.org/guide/topics/python-nonstandard-libraries/twython-guide/twitter-twython-simple-markov-bot.html
# http://tetration.xyz/Ngram-Tutorial/
# http://theorangeduck.com/page/17-line-markov-chain (Gerar texto)
# https://github.com/jsvine/markovify
# usa o arquivo SherlockHolmes.txt

import markovify
import re
from unicodedata import normalize

# Get raw text as string.
# with open("bases/Iracema-jose-de-alencar.txt") as f:
with open("SherlockHolmes.txt") as f:
    text = f.read()

#Remover acentos
def remover_acentos(txt, codif='latin-1'):   # 'utf-8'
    return normalize('NFKD', txt.decode(codif)).encode('ASCII', 'ignore'
        
text = remover_acentos(text)

# Build the model.
text_model = markovify.Text(text)

# Print five randomly-generated sentences
for i in range(5):
    print(text_model.make_sentence())

# Print three randomly-generated sentences of no more than 140 characters
for i in range(3):
    print(text_model.make_short_sentence(140))

#################	
# Outro exemplo #
#################
corpus = open("SherlockHolmes.txt").read()

text_model = markovify.Text(corpus, state_size=3)
model_json = text_model.to_json()
# In theory, here you'd save the JSON to disk, and then read it back later.

reconstituted_model = markovify.Text.from_json(model_json)
reconstituted_model.make_short_sentence(140)

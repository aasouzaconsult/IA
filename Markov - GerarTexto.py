# Livros - http://www.visionvox.com.br/busca.php?pagina=Nao&busca=A+viuvinha+Jos%E9+de+alencar&buscar=Buscar
# http://2017.compciv.org/guide/topics/python-nonstandard-libraries/twython-guide/twitter-twython-simple-markov-bot.html
# http://tetration.xyz/Ngram-Tutorial/
# http://theorangeduck.com/page/17-line-markov-chain (Gerar texto)
# https://github.com/jsvine/markovify
# usa o arquivo SherlockHolmes.txt

import markovify
import re
from unicodedata import normalize

# Get raw text as string.
# with open("SherlockHolmes.txt") as f:
# with open("bases/Iracema-jose-de-alencar.txt") as f:
with open("bases/Textos-Jose-De-Alencar.txt") as f:
    text = f.read()

#Remover acentos
def remover_acentos(txt, codif='latin-1'):   # 'utf-8'
    return normalize('NFKD', txt.decode(codif)).encode('ASCII', 'ignore'
        
text = remover_acentos(text)

# Build the model.
text_model = markovify.Text(text)

# Print five randomly-generated sentences
for i in range(10):
    print(text_model.make_sentence())

# Print three randomly-generated sentences of no more than 140 characters
for i in range(10):
    print(text_model.make_short_sentence(140))

#################	
# Outro exemplo #
#################
import re
from sys import stdout
from random import randint, choice
from collections import defaultdict

with open('bases/Textos-Jose-De-Alencar.txt') as f:
    words = remover_acentos(f)
    words = re.split(' +', f.read())

transition = defaultdict(list)
for w0, w1, w2 in zip(words[0:], words[1:], words[2:]):
    transition[w0, w1].append(w2)
	
i = randint(0, len(words)-3)
w0, w1, w2 = words[i:i+3]
for _ in range(500):
    stdout.write(w2+' ')
    w0, w1, w2 = w1, w2, choice(transition[w1, w2])

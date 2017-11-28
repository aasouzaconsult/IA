import re
import obo
from unicodedata import normalize

def remover_acentos(txt, codif='latin-1'):
    return normalize('NFKD', txt.decode(codif)).encode('ASCII', 'ignore')

# Texto Completo
with open("bases/Iracema-jose-de-alencar.txt") as f:
	text = f.read()

text = remover_acentos(text)

# Contar quantidade de espaços
len(re.findall('\s+', text))

# Capitulo I
with open("bases/Iracema-jose-de-alencar-Cap1.txt") as c1:
	Cap1 = c1.read()

Cap1 = remover_acentos(Cap1)
	
# Remove os espaços em branco
# Capítulo I
Cap1SE = re.sub(r'\s', '', Cap1)

# N-Gram (Treinamento?)
allMyWords = Cap1.split()
Cap1_nGram = obo.getNGrams(allMyWords, 3) # TriGram
print(obo.getNGrams(allMyWords, 3))

# N-Gram por caractere
def word2ngrams(text, n=1, exact=True):
  """ Convert text into character ngrams. """
  return ["".join(j) for j in zip(*[text[i:] for i in range(n)])]

word2ngrams(Cap1)

# Usando Markov
import random
class MarkovChain:
    def __init__(self):
        self.memory = {}
    def _learn_key(self, key, value):
        if key not in self.memory:
            self.memory[key] = []
        self.memory[key].append(value)
    def learn(self, text):
        tokens = text.split(" ")
        bigrams = [(tokens[i], tokens[i + 1]) for i in range(0, len(tokens) - 1)]
        for bigram in bigrams:
            self._learn_key(bigram[0], bigram[1])
    def _next(self, current_state):
        next_possible = self.memory.get(current_state)
        if not next_possible:
            next_possible = self.memory.keys()
        return random.sample(next_possible, 1)[0]
    def babble(self, amount, state=''):
        if not amount:
            return state
        next_word = self._next(state)
        return state + ' ' + self.babble(amount - 1, next_word)

# Instanciando Markov
m = MarkovChain()
m.learn(Cap1)
print(m.memory)

# Voltar os espaços


# Referências
# https://sookocheff.com/post/nlp/ngram-modeling-with-markov-chains/

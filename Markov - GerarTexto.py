# https://github.com/jsvine/markovify
# usa o arquivo SherlockHolmes.txt 

import markovify

# Get raw text as string.
with open("SherlockHolmes.txt") as f:
    text = f.read()

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

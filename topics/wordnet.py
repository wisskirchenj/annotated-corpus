import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('omw-1.4')

print("Hypernyms :  ", wn.synset('building.n.01').hypernyms())
from spacy import load

nlp = load('en_core_web_sm')
with open('../data/hyperskill-dataset-97961053.txt', 'r', encoding="utf-8") as file:
    content = file.read()
# doc = nlp('There are two error measurements in stemming algorithms, overstemming and understemming')
doc = nlp(content)
print(' '.join([token.lemma_ for token in doc]))
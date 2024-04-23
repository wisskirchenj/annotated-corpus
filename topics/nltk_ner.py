import nltk
from nltk import word_tokenize, pos_tag

nltk.download('words')
nltk.download('maxent_ne_chunker')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
tagged = pos_tag(word_tokenize("On March 1, 1932, Puyi was installed by the Japanese as the ruler of Manchukuo, considered by most historians as a puppet state of Imperial Japan, under the reign title Datong."))

result = [chunk[0] for chunk in nltk.ne_chunk(tagged) if hasattr(chunk, 'label') and chunk.label() == 'GPE']
print(result)
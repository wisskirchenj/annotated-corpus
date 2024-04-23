import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

text = """CONSIDERING that on 29 March 2017 the United Kingdom of Great Britain and Northern Ireland ("United Kingdom"), following the outcome of a referendum held in the United Kingdom and its sovereign decision to leave the European Union, notified its intention to withdraw from the European Union ("Union") and the European Atomic Energy Community ("Euratom") in accordance with Article 50 of the Treaty on European Union ("TEU"), which applies to Euratom by virtue of Article 106a of the Treaty establishing the European Atomic Energy Community ("Euratom Treaty")"""

doc = nlp(text)  # we use a part of the Brexit agreement
displacy.serve(doc, style="ent", auto_select_port=True)

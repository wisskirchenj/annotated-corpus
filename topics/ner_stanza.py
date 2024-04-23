import stanza
stanza.download('en')
nlp = stanza.Pipeline(lang="en")

doc = nlp('In May 1988, the Atlantic Records held a 40th Anniversary concert, broadcast on HBO. This concert, '
          'which was almost 13 hours in length, featured performances by a large number of their artists and included '
          'reunions of some rock legends like Led Zeppelin and Crosby, Stills, and Nash')

result = [ent.text for ent in doc.ents if ent.type == 'ORG']
print(result)

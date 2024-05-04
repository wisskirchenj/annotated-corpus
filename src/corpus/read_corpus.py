import re
import spacy
import numpy as np
import pandas as pd
from corpus.get_stats import get_stats

nlp = spacy.load('en_core_web_sm')


class Corpus:
    def __init__(self):
        self.data = np.array(['Token', 'Lemma', 'POS', 'Entity_type', 'IOB_tag'])
        self.stats = dict()

    @staticmethod
    def read_corpus(path: str) -> str:
        with open(path, 'r', encoding="utf-8") as file:
            return file.read()

    def add_nlp_attributes(self, content: str):
        doc = nlp(content)
        self.stats['Number of multi-word named entities'] = len([ent for ent in doc.ents if ' ' in ent.text])
        doc = [token for token in doc if not re.search(r'[><_\\/*\s]', token.text)]
        for token in doc:
            self.data = np.vstack(
                [self.data, (token.text, token.lemma_, token.pos_, token.ent_type_, token.ent_iob_)])

    def output_stat(self):
        # output all keys and values in stats
        for key, value in self.stats.items():
            print(f"{key}: {value}")

    def main(self):
        path = input()
        content = self.read_corpus(path)
        self.add_nlp_attributes(content)
        self.stats = get_stats(pd.DataFrame(self.data[1:], columns=self.data[0]), self.stats)
        self.output_stat()


if __name__ == '__main__':
    Corpus().main()

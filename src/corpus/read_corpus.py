import re
import spacy
import numpy as np
import pandas as pd

nlp = spacy.load('en_core_web_sm')


class Corpus:
    def __init__(self):
        self.data = np.array(['Token', 'Lemma', 'POS', 'Entity_type', 'IOB_tag'])

    @staticmethod
    def read_corpus(path: str) -> str:
        with open(path, 'r', encoding="utf-8") as file:
            return file.read()

    def add_lemma_and_pos(self, content: str):
        doc = nlp(content)
        doc = [token for token in doc if not re.search(r'[><_\\/*\s]', token.text)]
        for token in doc:
            self.data = np.vstack(
                [self.data, (token.text, token.lemma_, token.pos_, token.ent_type_, token.ent_iob_)])

    def output_as_dataframe(self):
        df = pd.DataFrame(self.data[1:], columns=self.data[0])
        print(df.head(20))

    def main(self):
        path = input()
        content = self.read_corpus(path)
        self.add_lemma_and_pos(content)
        self.output_as_dataframe()


if __name__ == '__main__':
    Corpus().main()

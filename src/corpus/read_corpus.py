import re
import spacy
import numpy as np

nlp = spacy.load('en_core_web_sm')


class Corpus:
    def __init__(self):
        self.data = np.array(['Token'])

    @staticmethod
    def read_corpus(path: str) -> str:
        with open(path, 'r', encoding="utf-8") as file:
            return file.read()

    def add_lemma_and_pos(self, content: str):
        for col_name in ['Lemma', 'POS']:
            self.data = np.hstack([self.data, np.array([col_name])])
        doc = nlp(content)
        doc = [token for token in doc if not re.search(r'[><_\\/*\s]', token.text)]
        for token in doc:
            self.data = np.vstack([self.data, (token.text, token.lemma_, token.pos_)])

    def output(self):
        print(self.data)
        print('Number of tokens: ', len(self.data))

    def main(self):
        path = input()
        content = self.read_corpus(path)
        self.add_lemma_and_pos(content)
        self.output()


if __name__ == '__main__':
    Corpus().main()

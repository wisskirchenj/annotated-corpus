import spacy
import numpy as np

nlp = spacy.load('en_core_web_sm')


class Corpus:
    def __init__(self):
        self.data = np.array(['Token'])

    def read_corpus(self, path: str):
        with open(path, 'r', encoding="utf-8") as file:
            content = file.read()
        doc = nlp(content)
        for token in doc:
            self.data = np.vstack([self.data, token.text])

    def output(self):
        print(self.data)
        print('Number of tokens: ', len(self.data))

    def main(self):
        path = input()
        self.read_corpus(path)
        self.output()


if __name__ == '__main__':
    Corpus().main()

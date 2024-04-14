import unittest
import pytest

from corpus.read_corpus import Corpus


class TestReadCorpus(unittest.TestCase):

    #  Reads a file from a given path and stores its content in a variable
    def test_read_file_and_store_content_tokens(self):
        corpus = Corpus()
        corpus.read_corpus('test.txt')
        assert 'Satz' in corpus.data
        assert 'Dies' in corpus.data
        assert 'Token' in corpus.data

    #  Tokenizes the content using Spacy's nlp object
    def test_tokenize_content(self):
        corpus = Corpus()
        corpus.read_corpus('test.txt')
        assert len(corpus.data) == 5

    #  reads an empty file and returns an empty numpy array
    def test_empty_file_returns_empty_array(self):
        corpus = Corpus()
        corpus.read_corpus('empty_file.txt')
        assert len(corpus.data) == 1

    #  raises a FileNotFoundError
    def test_raises_file_not_found_error(self):
        corpus = Corpus()
        with pytest.raises(FileNotFoundError):
            corpus.read_corpus('nonexistent_file.txt')

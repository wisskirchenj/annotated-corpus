import unittest
import pytest

from corpus.read_corpus import Corpus


class TestReadCorpus(unittest.TestCase):

    #  Reads a file from a given path
    def test_read_file_and_store_content_tokens(self):
        corpus = Corpus()
        corpus.read_corpus('test/test.txt')
        self.assertEqual('Dies ist ein Satz', corpus.read_corpus('test/test.txt'))

    #  Tokenizes the content using Spacy's nlp object
    def test_tokenize_content(self):
        corpus = Corpus()
        corpus.add_lemma_and_pos(corpus.read_corpus('test/test.txt'))
        assert len(corpus.data) == 5

    #  Tokenizes a file with special characters
    def test_tokenize_filtered_content(self):
        corpus = Corpus()
        corpus.add_lemma_and_pos(corpus.read_corpus('test/test_with_specials.txt'))
        self.assertEqual(10, len(corpus.data))
        assert 'This' in corpus.data
        assert '.' in corpus.data
        assert 'PUNCT' in corpus.data
        assert 'NOUN' in corpus.data
        assert 'be' in corpus.data

    #  reads an empty file and returns an empty numpy array
    def test_empty_file_returns_empty_array(self):
        corpus = Corpus()
        corpus.read_corpus('test/empty_file.txt')
        assert len(corpus.data) == 1

    #  raises a FileNotFoundError
    def test_raises_file_not_found_error(self):
        corpus = Corpus()
        with pytest.raises(FileNotFoundError):
            corpus.read_corpus('nonexistent_file.txt')

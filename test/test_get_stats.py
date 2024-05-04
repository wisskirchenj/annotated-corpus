from pandas import DataFrame

from corpus.get_stats import find_most_common_non_english_words, find_most_common_named_entity


class TestGetStats:

    #  Verify that the function returns a dictionary.
    def test_returns_dictionary(self):
        df = DataFrame({'Lemma': ['apple', 'banana', 'orange'], 'POS': ['NOUN', 'NOUN', 'NOUN']})
        result = find_most_common_non_english_words(df, 10)
        assert isinstance(result, dict)

    #  Verify that the dictionary returned by the function contains the expected keys.
    def test_contains_expected_keys(self):
        df = DataFrame({'Lemma': ['pommes', 'tomate', 'orange', 'gün'], 'POS': ['NOUN', 'NOUN', 'NOUN', 'NOUN']})
        result = find_most_common_non_english_words(df, 10)
        expected_keys = {'pommes', 'tomate'}
        assert set(result.keys()) == expected_keys

    #  Verify that the function returns an empty dictionary if there are no non-English words in the DataFrame.
    def test_no_non_english_words(self):
        df = DataFrame({'Lemma': ['apple', 'banana', 'orange'], 'POS': ['NOUN', 'NOUN', 'NOUN']})
        result = find_most_common_non_english_words(df, 10)
        assert result == {}

    #  Returns a string with the most common named entity and its entity type in the DataFrame
    def test_returns_most_common_named_entity(self):
        df = DataFrame({'Lemma': ['apple', 'banana', 'apple', 'orange'],
                        'Entity_type': ['fruit', '', 'fruit', '']})
        result = find_most_common_named_entity(df)
        assert result == "('apple', 'fruit')"

    #  Returns the first most common named entity and its entity type when there are ties
    def test_returns_first_most_common_named_entity_with_ties(self):
        df = DataFrame({'Lemma': ['apple', 'banana', 'banana', 'orange'],
                        'Entity_type': ['fruit', '', 'fruit', '']})
        result = find_most_common_named_entity(df)
        assert result == "('apple', 'fruit')"

    def test_only_entities_counted(self):
        df = DataFrame({'Lemma': ['apple', 'banana', 'banana', 'banana', 'apple'],
                        'Entity_type': ['fruit', '', 'fruit', '', 'fruit']})
        result = find_most_common_named_entity(df)
        assert result == "('apple', 'fruit')"

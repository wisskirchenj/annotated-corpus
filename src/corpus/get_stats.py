from nltk.corpus import words
from pandas import DataFrame
from scipy.stats import pearsonr


def find_most_common_non_english_words(df: DataFrame, count: int) -> dict:
    english_words = words.words()
    pos_criteria = {'NOUN', 'VERB', 'ADJ', 'ADV'}
    query = (df['Lemma'].str.len() > 4) & df['POS'].isin(pos_criteria) & ~df['Lemma'].str.lower().isin(english_words)
    return df[query]['Lemma'].value_counts().head(count).to_dict()


def find_most_common_named_entity(df: DataFrame) -> str:
    named_ents = df[df['Entity_type'] != '']
    most_common = named_ents['Lemma'].mode()[0]
    entity_type = named_ents.loc[named_ents['Lemma'] == most_common, 'Entity_type'].mode()[0]
    return f"('{most_common}', '{entity_type}')"


def correlate_noun_propn_and_named_entities(df: DataFrame) -> str:
    correlation = pearsonr(df['POS'].isin(['NOUN', 'PROPN']), df['Entity_type'] != '')[0]
    return f"{correlation:.2f}"


def get_stats(df: DataFrame, stats: dict) -> dict:
    stats["Number of lemmas 'devotchka'"] = df[df['Lemma'] == 'devotchka'].shape[0]
    stats["Number of tokens with the stem 'milk'"] = df[df['Lemma'].str.lower().str.contains('milk')].shape[0]
    stats["Most frequent entity type"] = df[df['Entity_type'] != '']['Entity_type'].mode()[0]
    stats["Most frequent named entity token"] = find_most_common_named_entity(df)
    stats["Most common non-English words"] = find_most_common_non_english_words(df, 10)
    stats["Correlation between NOUN and PROPN and named entities"] = correlate_noun_propn_and_named_entities(df)
    return stats

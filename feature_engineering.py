import spacy
import yaml
from statistics import mean
import pandas as pd

nlp = spacy.load('en_core_web_sm')


def read_yaml(yaml_filepath):
    with open(yaml_filepath, 'r') as infile:
        language_list_dict = yaml.load(infile, Loader=yaml.FullLoader)
    return language_list_dict


def count_word_groups(text, word_list):
    word_count = 0

    for token in text:
        if token in word_list:
            word_count += 1

    return word_count


def add_feature_columns(df):
    """
    Adds features to existing df. Requires 'text' and 'speaker' column. Adds following columns: first name, last name,
    token list, lemma list, pos list, adverb count, several length measures, unique word ratio and counts for specific
    words.

    :param df: pd.DataFrame object with columns "text" and "speaker"
    :return: df
    """
    # Count specific groups
    yaml_filepath = './language_lists.yaml'
    word_groups_dict = read_yaml(yaml_filepath)


    # Split name
    df['first_name'] = df.apply(lambda row: row['speaker'].split(' ')[0], axis=1)
    df['last_name'] = df.apply(lambda row: row['speaker'].split(' ')[1], axis=1)

    # Get spacy info
    df['spacy_type'] = df.apply(lambda row: [token for token in nlp(row['text'].strip())], axis=1)
    df['tokens'] = df.apply(lambda row: [token.text for token in row['spacy_type']], axis=1)
    df['lemmas'] = df.apply(lambda row: [token.lemma_ for token in row['spacy_type']], axis=1)
    df['pos'] = df.apply(lambda row: [token.pos for token in row['spacy_type']], axis=1)

    # General feats
    df['adv_count'] = df.apply(lambda row: row['pos'].count(86), axis=1)

    df['NumberOfCharacters'] = df.apply(lambda row: len(row['text']), axis=1)
    df['MeanCharactersPerWord'] = df.apply(lambda row: mean(len(token) for token in row['tokens']), axis=1)

    df['SentenceLength'] = df.apply(lambda row: len(row['tokens']), axis=1)
    df['NumberofUniqueWords'] = df.apply(lambda row: len(set(row['tokens'])), axis=1)
    df['WordsVsUnique'] = df.apply(lambda row: row['NumberofUniqueWords'] / row['SentenceLength'], axis=1)

    for key in word_groups_dict:
        df[f'Count_{key}'] = df.apply(lambda row: count_word_groups(row['lemmas'], word_groups_dict[key]), axis=1)

    df.drop(['spacy_type'], axis=1, inplace=True)

    return df


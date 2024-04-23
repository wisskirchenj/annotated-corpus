# NLP Corpus annotation

Project implemented in the track 'Python for Science' of hyperskill.org JetBrains Academy. 
Purpose of doing this project is to make progress in this track and refresh some Python.

## Technology / External Libraries

- Python 3.12.3,
- with packages: numpy, spacy, pandas, unittest, unittest.mock, parameterized

## Repository Contents

The sources of main project tasks (4 stages) and pytest testing.

## Installation

Install the wheel file using pip after clone/fork:

    > pip install dist/annotated_corpus-3.12.3-py3-none-any.whl

## Program description

Annotated corpus is a Python-based NLP project. Any serious NLP experiment requires data processing. In most cases, you
use ready-made data, but sometimes you need to compile a corpus yourself. Depending on the task, you may also need
certain information about your text: part-of-speech tags, named entities, statistical characteristics, and so on. In
this project, raw text is converted into a corpus for further research.

Have fun!

## Project progress

[//]: # (Project was completed on 29.10.23)

13.04.24 Project started and Stage 1 completed. Spacy with en_core_web_sm model is used to tokenize a text and store
tokens in a numpy array.

23.04.24 
- Stage 2 completed. Add lemmatisation and POS tagging to the tokens. Preprocess tokens by filtering out.
- Stage 3 completed. Add named entity recognition to the tokens. Transform ndarray to a pandas dataframe.

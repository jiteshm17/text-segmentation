import nltk.data
import numpy as np
from nltk.tokenize import RegexpTokenizer
import wiki_utils
import wiki_thresholds
import utils
import logging

# Initialize global variables
sentence_tokenizer = None
words_tokenizer = None
missing_stop_words = {'of', 'a', 'and', 'to'}
logger = utils.setup_logger(__name__, 'text_manipulation.log', delete_old=True)


def get_punkt():
    global sentence_tokenizer
    if sentence_tokenizer:
        return sentence_tokenizer

    try:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    except LookupError:
        nltk.download('punkt')
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentence_tokenizer = tokenizer
    return sentence_tokenizer

def get_words_tokenizer():
    global words_tokenizer
    if words_tokenizer:
        return words_tokenizer

    words_tokenizer = RegexpTokenizer(r'\w+')
    return words_tokenizer

def split_sentence_with_list(sentence):
    list_pattern = "\n" + wiki_utils.get_list_token() + "."
    if sentence.endswith(list_pattern):
        split_sentence = [s for s in sentence.split(list_pattern) if len(s) > 0]
        split_sentence.append(wiki_utils.get_list_token() + ".")
        return split_sentence
    else:
        return [sentence]

def split_sentence_colon_new_line(sentence):
    split_sentence = sentence.split(":\n")
    if len(split_sentence) == 1:
        return split_sentence
    
    new_sentences = []
    for i in range(len(split_sentence) - 1):
        if len(split_sentence[i]) > 0:
            new_sentences.append(split_sentence[i] + ":")
    
    if len(split_sentence[-1]) > 0:
        new_sentences.append(split_sentence[-1])
    
    return new_sentences

def split_long_sentences_with_backslash_n(max_words_in_sentence, sentences, doc_id):
    new_sentences = []
    for sentence in sentences:
        sentence_words = extract_sentence_words(sentence)
        if len(sentence_words) > max_words_in_sentence:
            split_sentences = sentence.split('\n')
            if len(split_sentences) > 1:
                logger.info(f"Sentence with backslash was split. Doc Id: {doc_id}   Sentence: {sentence}")
            new_sentences.extend(split_sentences)
        else:
            if "\n" in sentence:
                logger.info(f"No split for sentence with backslash n. Doc Id: {doc_id}   Sentence: {sentence}")
            new_sentences.append(sentence)
    return new_sentences

def split_sentences(text, doc_id):
    sentences = get_punkt().tokenize(text)
    sentences_list_fixed = []
    for sentence in sentences:
        split_list_sentence = split_sentence_with_list(sentence)
        sentences_list_fixed.extend(split_list_sentence)

    sentences_colon_fixed = []
    for sentence in sentences_list_fixed:
        split_colon_sentence = split_sentence_colon_new_line(sentence)
        sentences_colon_fixed.extend(split_colon_sentence)

    sentences_no_backslash_n = split_long_sentences_with_backslash_n(
        wiki_thresholds.max_words_in_sentence_with_backslash_n, 
        sentences_colon_fixed, 
        doc_id
    )

    ret_sentences = [sentence.replace('\n', ' ') for sentence in sentences_no_backslash_n]
    return ret_sentences

def extract_sentence_words(sentence, remove_missing_emb_words=False, remove_special_tokens=False):
    if remove_special_tokens:
        for token in wiki_utils.get_special_tokens():
            sentence = sentence.replace(token, "")
    
    tokenizer = get_words_tokenizer()
    sentence_words = tokenizer.tokenize(sentence)
    
    if remove_missing_emb_words:
        sentence_words = [w for w in sentence_words if w not in missing_stop_words]

    return sentence_words

def word_model(word, model):
    if model is None:
        return np.random.randn(1, 300)
    else:
        if word in model:
            return model[word].reshape(1, 300)
        else:
            # If word not in model, return 'UNK' embedding
            return model['UNK'].reshape(1, 300)
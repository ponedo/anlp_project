import nltk
import os
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk import tree2conlltags, conlltags2tree
from nltk import ne_chunk
from nltk import RegexpParser

data_dir = "..\\ivan_data\\merged"
file_list = os.listdir(data_dir)

ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'


def read_data(data_dir, data_file_name):
    return open(os.path.join(data_dir, data_file_name), "r", encoding="cp950").read()


def preprocess(raw_text):
    sents = sent_tokenize(raw_text)
    sents = [pos_tag(word_tokenize(sent)) for sent in sents]
    return sents


# use regex rule to parse
def regex_parse(pattern, pos_tagged_sent):
    parser = RegexpParser(pattern)
    parse_tree = parser.parse(pos_tagged_sent)
    parse_text = tree2conlltags(parse_tree)
    only_parse_text = [t for t in parse_text if t[2] != "O"]
    only_parse_tree = conlltags2tree(only_parse_text)
    return only_parse_tree


# using nltk ne_chunk to extract NE
def ne_extract(pos_tagged_sent):
    ne_tree = ne_chunk(pos_tagged_sent)
    ne_text = tree2conlltags(ne_tree)
    only_ne_text = [t for t in ne_text if t[2] != "O"]
    only_ne_tree = conlltags2tree(only_ne_text)
    return only_ne_tree


def pipeline(file_id):
    try:
        raw_text = read_data(data_dir, file_list[file_id])
    except:
        raise Exception("File not found!")
    sents = preprocess(raw_text)
    fo = open("nltk_ner_out_{}".format(file_id), "w")
    for sent in sents:
        ne = ne_extract(sent)
        if ne:
            # print(ne)
            fo.write(str(ne))
    fo.close()
    
import os
import collections
import string
from nltk.stem.snowball import SnowballStemmer
from nltk import RegexpParser


corpus_dir = r"G:\\gmb-2.2.0"


def regex_parse(pattern, pos_tagged_sent):
    parser = RegexpParser(pattern)
    parse_tree = parser.parse(pos_tagged_sent)
    parse_text = tree2conlltags(parse_tree)
    only_parse_text = [t for t in parse_text if t[2] != "O"]
    only_parse_tree = conlltags2tree(only_parse_text)
    return only_parse_tree


def features(tokens, index, history):
    """
    tokens: a POS-tagged sentence [(w0, t0), (w1, t1), ...]
    index:  the index of the token we want to extract features for
    history: the previous predicted IOB tags
    """

    # init the stemmer
    stemmer = SnowballStemmer('english')

    # Pad the sequence with placeholders
    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + \
        list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
    history = ['[START2]', '[START1]'] + list(history)

    # shift the index with 2, to accommodate the padding
    index += 2

    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = history[index - 1]

    # return a dict of features
    return {
        'word': word, 
        'lemma': stemmer.stem(word), 
        'pos': pos, 
        'all_ascii': all([ch in string.ascii_lowercase for ch in word]), 

        'next_word': nextword, 
        'next_lemma': stemmer.stem(nextword), 
        'next_pos': nextpos, 

        "prev_word": prevword, 
        "prev_lemma": stemmer.stem(prevword), 
        "prev_pos": prevpos, 

        "prev_prev_word": prevprevword, 
        "prev_prev_pos": prevprevpos, 
        
        "prev_iob": previob, 

        "contains_dash": '-' in word, 
        "contains_dot": '.' in word, 

        "all_caps": word == word.capitalize(), 
        "caplitalized": word[0] in string.ascii_uppercase, 
        
        "prev_all_caps": prevword == prevword.capitalize(), 
        "prev_capitalized": prevword[0] in string.ascii_uppercase, 

        "next_all_caps": nextword == nextword.capitalize(), 
        "next_capitalized": nextword[0] in string.ascii_uppercase
    }


def to_conll_iob(annotated_sent):
    """
    annotated_sent: list of triplets [(w0, pos0, ner0), (w1, pos1, ner1), ...]
    
    Transform a pseudo-IOB notation: O, PERSON, PERSON, O, O, LOCATION, O
    to proper IOB notation: O, B-PERSON, I-PERSON, O, O, B-LOCATION, O
    """
    proper_iob_tokens = []
    for idx, annotated_token in enumerate(annotated_sent):
        tag, word, ner = annotated_token
        if ner != "O":
            if idx == 0:
                ner = "B-" + ner
            elif annotated_sent[idx -1][2] == ner:
                ner = "I-" + ner
            else:
                ner = "B-" + ner
        proper_iob_tokens.append((tag, word, ner))
    return proper_iob_tokens


def read_gmb(corpus_dir):
    for root, _, files in os.walk(corpus_dir):
        for filename in files:
            if filename.endswith(".tags"):
                with open(os.path.join(root, filename), 'rb') as f:
                    content = f.read().decode('utf-8').strip()
                    sents = content.split('\n\n')
                    for sent in sents:
                        tokens = [seq for seq in sent.split('\n') if seq]
                        standard_form_tokens = []
                        
                        for token in tokens:
                            annotations = token.split('\t')
                            word, pos, ner = annotations[0], annotations[1], annotations[3]
                            if ner != "O":
                                ner = ner.split('-')[0]
                            if pos in ('LQU', 'RQU'):   # Make it NLTK compatible
                                pos = "``"
                            standard_form_tokens.append((word, pos, ner))

                        conll_tokens = to_conll_iob(standard_form_tokens)

                        # Make it NLTK Classifier compatible - [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
                        # Because the classfier expects a tuple as input, first item input, second the class
                        yield [((w, t), iob) for w, t, iob in conll_tokens]

reader = read_gmb(corpus_dir)


import pickle
from nltk.tag import ClassifierBasedTagger
from nltk.chunk import ChunkParserI
from nltk import conlltags2tree, tree2conlltags


class NamedEntityChunker(ChunkParserI):
    def __init__(self, train_sents, **kwargs):
        self.feature_dectector = features
        self.tagger = ClassifierBasedTagger(
            train=train_sents, 
            feature_detector=features, 
            **kwargs
        )

    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)

        # Transform the result from [((w1, t1), iob1), ...] 
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]

        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)

data = list(reader)
training_samples = data[:int(len(data) * 0.9)]
test_samples = data[int(len(data) * 0.9):]
 
print("#training samples = %s" % len(training_samples)) # training samples = 55809
print("#test samples = %s" % len(test_samples)) # test samples = 6201

chunker = NamedEntityChunker(training_samples[:2000])

from nltk import pos_tag, word_tokenize
print(chunker.parse(pos_tag(word_tokenize("I'm going to Germany this Monday."))))
"""
(S
  I/PRP
  'm/VBP
  going/VBG
  to/TO
  (geo Germany/NNP)
  this/DT
  (tim Monday/NNP)
  ./.)
"""

score = chunker.evaluate([conlltags2tree([(w, t, iob) for (w, t), iob in iobs]) for iobs in test_samples[:500]])
print(score.accuracy()) # 0.931132334092 - Awesome :D
import nltk
import os
import pickle
from nltk import pos_tag
from nltk import TaggerI
from pprint import pprint
import en_core_web_sm
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from itertools import islice

###########
# taggers #
###########

class MyTagger(TaggerI):
    """
    Tagging pos for a sentence using nltk.pos_tag
    """
    def tag(self, tokens):
        """
        Determine the most appropriate tag sequence for the given
        token sequence, and return a corresponding list of tagged
        tokens.  A tagged token is encoded as a tuple ``(token, tag)``.

        :rtype: list(tuple(str, str))
        """
        return pos_tag(tokens)

    def evaluate(self, gold):
        """
        gold: list(list(tuple(w, t)))
        """
        correct, total = 0, 0
        for gold_sent in gold:
            tokens = []
            gold_tags = []
            for token, tag in gold_sent:
                tokens.append(token)
                if '-' in tag:
                    tag = tag.split('-')[0]
                gold_tags.append(tag)
            pred_tags = [tag for token, tag in self.tag(tokens)]
            if len(pred_tags) != len(gold_tags):
                continue
            for pred_tag, gold_tag in zip(pred_tags, gold_tags):
                total += 1
                if pred_tag == gold_tag:
                    correct += 1
        return correct / total


class SpacyTagger(MyTagger):
    """
    Tagging pos with spacy pos tagging function
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.nlp = en_core_web_sm.load()

    def tag(self, tokens):
        text = self.nlp(" ".join(tokens))
        return [(token.text, token.tag_) for token in text]


def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }


class DecisionTreeTagger(TaggerI):
    @classmethod
    def transform_to_dataset(cls, tagged_sentences):
        X, y = [], []
    
        for tagged in tagged_sentences:
            for index in range(len(tagged)):
                tokens = [w for w, t in tagged]
                X.append(features(tokens, index))
                y.append(tagged[index][1])
    
        return X, y

    def __init__(self):
        self.classifier_ = Perceptron(verbose=10, n_jobs=-1, n_iter=5)
        # self.classifier_ = DecisionTreeClassifier(criterion="entropy")
        self.vectorizer_ = DictVectorizer(sparse=False)
        self.clf = Pipeline([
            ('vectorizer', self.vectorizer_), 
            ('classifier', self.classifier_)
        ])

    def train(self, tagged_sentences, batch_size=500):
        tagged_sentences = iter(tagged_sentences)
        # all_classes = [
        #     'NNP', ',', 'CD', 'NNS', 'JJ', 'MD', 'VB', 
        #     'DT', 'NN', 'IN', '.', 'VBZ', 'VBG', 'CC', 
        #     'VBD', 'VBN', '-NONE-', 'RB', 'TO', 'PRP', 
        #     'RBR', 'WDT', 'VBP', 'RP', 'PRP$', 'JJS', 
        #     'POS', '``', 'EX', "''", 'WP', ':', 'JJR', 
        #     'WRB', '$', 'NNPS', 'WP$', '-LRB-', '-RRB-', 
        #     'PDT', 'RBS', 'FW', 'UH', 'SYM', 'LS', '#']

        all_classes = [
            'NNS', 'IN', 'VBP', 'VBN', 'NNP', 'TO', 'VB',
            'DT', 'NN', 'CC', 'JJ', '.', 'VBD', 'WP',
            '``', 'CD', 'PRP', 'VBZ', 'POS', 'VBG',
            'RB', ',', 'WRB', 'PRP$', 'MD', 'WDT',
            'JJR', ':', 'JJS', 'WP$', 'RP', 'PDT',
            'NNPS', 'EX', 'RBS', 'LRB', 'RRB', '$',
            'RBR', ';', 'UH', 'FW']
        
        batch_num = 0
        batch = list(islice(tagged_sentences, batch_size))
        X, y = self.__class__.transform_to_dataset(batch)
        
        self.vectorizer_.fit(X)
        
        while len(X):
            print("\n\n************ Train Batch %d *************\n\n" %  batch_num)
            batch_num += 1
            X = self.vectorizer_.transform(X)
            self.classifier_.partial_fit(X, y, all_classes)
            batch = list(islice(tagged_sentences, batch_size))
            X, y = self.__class__.transform_to_dataset(batch)
    
        # self.clf.fit(X, y)
        save_tagger(self, "pos_tagger")
    
    def tag(self, words):
        tags = self.clf.predict(
            [features(words, index) for index in range(len(words))])
        return zip(words, tags)

    def evaluate(self, tagged_sentences, batch_size=500):
        tagged_sentences = iter(tagged_sentences)
        batch_num = 0
        correct, total = 0, 0

        batch = list(islice(tagged_sentences, batch_size))
        X, y = self.__class__.transform_to_dataset(batch)
        len_X = len(X)

        while len_X:
            print("\n\n************ Test Batch %d *************\n\n" %  batch_num)
            batch_num += 1
            
            correct += len_X * self.clf.score(X, y)
            total += len_X

            batch = list(islice(tagged_sentences, batch_size))
            X, y = self.__class__.transform_to_dataset(batch)
            len_X = len(X)

        return correct / total

###################
# corpus handling #
###################

def to_conll_iob(annotated_sent):
    """
    annotated_sent: list of triplets [(w0, pos0, ner0), (w1, pos1, ner1), ...]
    
    Transform a pseudo-IOB notation: O, PERSON, PERSON, O, O, LOCATION, O
    to proper IOB notation: O, B-PERSON, I-PERSON, O, O, B-LOCATION, O
    """
    proper_iob_tokens = []
    for idx, annotated_token in enumerate(annotated_sent):
        word, tag, ner = annotated_token
        if ner != "O":
            if idx == 0:
                ner = "B-" + ner
            elif annotated_sent[idx -1][2] == ner:
                ner = "I-" + ner
            else:
                ner = "B-" + ner
        proper_iob_tokens.append((word, tag, ner))
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

#######################
# training pos tagger #
#######################
gmb_train_len = 45000
corpus_dir = r"G:\\gmb-2.2.0"
reader_train = read_gmb(corpus_dir)

def train_sents_getter():
    for sent in reader_train:
        yield [token[0] for token in sent]
training_sentences = train_sents_getter()
training_sentences = islice(training_sentences, 45000)

def tags_getter():
    for sent in training_sentences:
        for w, t in sent:
            yield t
tags = tags_getter()

# tagged_sentences = nltk.corpus.treebank.tagged_sents()

# # Split the dataset for training and testing
# cutoff = int(.75 * len(tagged_sentences))
# training_sentences = tagged_sentences[:cutoff]
# test_sentences = tagged_sentences[cutoff:] 

####################
# evaluate taggers #
####################

# test_corpus = "treebank"
# test_corpus = "brown"
test_corpus = "gmb"

# tagger = MyTagger()
# tagger = SpacyTagger()
tagger = DecisionTreeTagger()

def test_tagger():
    if test_corpus in ["treebank", "brown"]:
        if test_corpus == "treebank":
            from nltk.corpus import treebank
            tagged_sents = treebank.tagged_sents()
            sents = treebank.sents()
        elif test_corpus == "brown":
            from nltk.corpus import brown
            tagged_sents = brown.tagged_sents()
            sents = brown.sents()

    elif test_corpus == "gmb":
        corpus_dir = r"G:\\gmb-2.2.0"
        reader = read_gmb(corpus_dir)
        data = islice(reader, 45000, 5000)
        sents = []
        tagged_sents = []
        for sent in data:
            token_sent = []
            tagged_token_sent = []
            for token in sent:
                token_sent.append(token[0][0])
                tagged_token_sent.append(token[0])
            sents.append(token_sent)
            tagged_sents.append(tagged_token_sent)

    print(tagger.evaluate(tagged_sents))

    # show test case (first sentence of the corpus)
    test_sent = sents[0]
    test_tagged_sent = tagged_sents[0]
    pprint([x for x in zip(tagger.tag(test_sent), test_tagged_sent)])

##################
# store and load #
##################

def save_tagger(tagger, file_name):
    with open(file_name + ".pk", "wb") as f:
        pickle.dump(tagger, f)


def load_tagger(file_name):
    with open(file_name + ".pk", "rb") as f:
        return pickle.load(f)
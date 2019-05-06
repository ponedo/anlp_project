import os
import re
import itertools
import pickle
from nltk import conlltags2tree 
from nltk import tree2conlltags
from nltk.chunk import ChunkParserI
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer


stemmer = SnowballStemmer('english') 
corpus_dir = r"G:\\gmb-2.2.0"


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
                        yield conlltags2tree(conll_tokens)

 
def shape(word):
    word_shape = 'other'
    if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word):
        word_shape = 'number'
    elif re.match('\W+$', word):
        word_shape = 'punct'
    elif re.match('[A-Z][a-z]+$', word):
        word_shape = 'capitalized'
    elif re.match('[A-Z]+$', word):
        word_shape = 'uppercase'
    elif re.match('[a-z]+$', word):
        word_shape = 'lowercase'
    elif re.match('[A-Z][a-z]+[A-Z][a-z]+[A-Za-z]*$', word):
        word_shape = 'camelcase'
    elif re.match('[A-Za-z]+$', word):
        word_shape = 'mixedcase'
    elif re.match('__.+__$', word):
        word_shape = 'wildcard'
    elif re.match('[A-Za-z0-9]+\.$', word):
        word_shape = 'ending-dot'
    elif re.match('[A-Za-z0-9]+\.[A-Za-z0-9\.]+\.$', word):
        word_shape = 'abbreviation'
    elif re.match('[A-Za-z0-9]+\-[A-Za-z0-9\-]+.*$', word):
        word_shape = 'contains-hyphen'
 
    return word_shape


def ner_features(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """
 
    # Pad the sequence with placeholders
    tokens = [('__START2__', '__START2__'), ('__START1__', '__START1__')] + list(tokens) + [('__END1__', '__END1__'), ('__END2__', '__END2__')]
    history = ['__START2__', '__START1__'] + list(history)
 
    # shift the index with 2, to accommodate the padding
    index += 2
 
    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = history[-1]
    prevpreviob = history[-2]
 
    feat_dict = {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
        'shape': shape(word),
 
        'next-word': nextword,
        'next-pos': nextpos,
        'next-lemma': stemmer.stem(nextword),
        'next-shape': shape(nextword),
 
        'next-next-word': nextnextword,
        'next-next-pos': nextnextpos,
        'next-next-lemma': stemmer.stem(nextnextword),
        'next-next-shape': shape(nextnextword),
 
        'prev-word': prevword,
        'prev-pos': prevpos,
        'prev-lemma': stemmer.stem(prevword),
        'prev-iob': previob,
        'prev-shape': shape(prevword),
 
        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,
        'prev-prev-lemma': stemmer.stem(prevprevword),
        'prev-prev-iob': prevpreviob,
        'prev-prev-shape': shape(prevprevword),
    }
 
    return feat_dict


class ScikitLearnChunker(ChunkParserI):
 
    @classmethod
    def to_dataset(cls, parsed_sentences, feature_detector):
        """
        Transform a list of tagged sentences into a scikit-learn compatible POS dataset
        :param parsed_sentences:
        :param feature_detector:
        :return:
        """
        X, y = [], []
        for parsed in parsed_sentences:
            iob_tagged = tree2conlltags(parsed)
            words, tags, iob_tags = zip(*iob_tagged)
 
            tagged = list(zip(words, tags))
 
            for index in range(len(iob_tagged)):
                X.append(feature_detector(tagged, index, history=iob_tags[:index]))
                y.append(iob_tags[index])
 
        return X, y
 
    @classmethod
    def get_minibatch(cls, parsed_sentences, feature_detector, batch_size=500):
        batch = list(itertools.islice(parsed_sentences, batch_size))
        X, y = cls.to_dataset(batch, feature_detector)
        return X, y
 
    @classmethod
    def train(cls, parsed_sentences, feature_detector, all_classes, **kwargs):
        batch_num = 0

        X, y = cls.get_minibatch(parsed_sentences, feature_detector, kwargs.get('batch_size', 500))
        vectorizer = DictVectorizer(sparse=False)
        vectorizer.fit(X)
 
        clf = Perceptron(verbose=10, n_jobs=-1, n_iter=kwargs.get('n_iter', 5))

        while len(X):
            print("\n\n************ Train Batch %d *************\n\n" %  batch_num)
            batch_num += 1
            X = vectorizer.transform(X)
            clf.partial_fit(X, y, all_classes)
            X, y = cls.get_minibatch(parsed_sentences, feature_detector, kwargs.get('batch_size', 500))
        
        clf = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', clf)
        ])

        return cls(clf, feature_detector)
 
    def __init__(self, classifier, feature_detector):
        self._classifier = classifier
        self._feature_detector = feature_detector
 
    def parse(self, tokens):
        """
        Chunk a tagged sentence
        :param tokens: List of words [(w1, t1), (w2, t2), ...]
        :return: chunked sentence: nltk.Tree
        """
        history = []
        iob_tagged_tokens = []
        for index, (word, tag) in enumerate(tokens):
            iob_tag = self._classifier.predict([self._feature_detector(tokens, index, history)])[0]
            history.append(iob_tag)
            iob_tagged_tokens.append((word, tag, iob_tag))
 
        return conlltags2tree(iob_tagged_tokens)
 
    def score(self, parsed_sentences):
        """
        Compute the accuracy of the tagger for a list of test sentences
        :param parsed_sentences: List of parsed sentences: nltk.Tree
        :return: float 0.0 - 1.0
        """
        batch_num = 0

        correct, total = 0, 0
        X_test, y_test = self.__class__.get_minibatch(parsed_sentences, self._feature_detector)
        X_len = len(X_test)
        
        while X_len:
            print("\n\n************ Test Batch %d *************\n\n" %  batch_num)
            batch_num += 1

            batch_accuracy = self._classifier.score(X_test, y_test)
            correct += batch_accuracy * X_len
            total += X_len
            X_test, y_test = self.__class__.get_minibatch(parsed_sentences, self._feature_detector)
            X_len = len(X_test)
        
        return correct / total

def train_perceptron():
    reader = read_gmb(corpus_dir)
 
    all_classes = ['O', 'B-per', 'I-per', 'B-gpe', 'I-gpe', 
                   'B-geo', 'I-geo', 'B-org', 'I-org', 'B-tim', 'I-tim',
                   'B-art', 'I-art', 'B-eve', 'I-eve', 'B-nat', 'I-nat']
 
    pa_ner = ScikitLearnChunker.train(
        itertools.islice(reader, 50000), feature_detector=ner_features,
        all_classes=all_classes, batch_size=500, n_iter=5)
    print("==================== Train over =====================")
    accuracy = pa_ner.score(itertools.islice(reader, 5000))
    print("Accuracy:", accuracy) # 0.970327096314
    save_tagger(pa_ner, "ne_tagger")
    return pa_ner


def save_tagger(tagger, file_name):
    with open(file_name + ".pk", "wb") as f:
        pickle.dump(tagger, f)


def load_tagger(file_name):
    with open(file_name + ".pk", "rb") as f:
        return pickle.load(f)
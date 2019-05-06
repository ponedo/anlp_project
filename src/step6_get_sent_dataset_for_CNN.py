import pickle
import re
import string
import os
import random
from utils import load_stuff,\
    spacy_docs_dir, spacy_docs_list, \
    load_sent_labels, filter_sent, \
    calc_nc_avg_tfidf, stopwords_set


def load_data_label_for_CNN(file_id):
    with open(os.path.join(spacy_docs_dir, spacy_docs_list[file_id]), "rb") as f:
        doc = pickle.load(f)
    sent_labels_doc = load_sent_labels(file_id)
    # print(sent_labels_doc)

    sent_num = sum([1 for t in doc.sents])
    for idx, sent in enumerate(doc.sents):
        if idx >= 0.4 * sent_num:
            break
        if filter_sent(sent):
            data = [token.lemma_ for token in sent if not (token.is_punct or token.lemma_ in stopwords_set)]
            label = sent_labels_doc[idx][1]
            # label = random.random() < 0.5 # randomly generate labels
            yield idx, data, label
            if label == 1.:
                for _ in range(60):
                    yield idx, data, label


def load_dataset_for_CNN(file_ids):
    for i in file_ids:
        file_name = spacy_docs_list[i]
        print(file_name)
        yield from load_data_label_for_CNN(i)
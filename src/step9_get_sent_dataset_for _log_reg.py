import pickle
import re
import string
import os
import random
from utils import load_stuff,\
    spacy_docs_dir, spacy_docs_list, \
    load_sent_labels, filter_sent, \
    calc_nc_avg_tfidf, stopwords_set
from step7_train_CNN import get_word_ids, vocab


tfidfs = load_stuff("tfidfs")
# try:
#     cnn_model = load_stuff("CNN")
#     cnn_model.load_weights("CNN.hdf5")
# except:
#     print("Warning: CNN model not trained yet!")


tags = [
    "-LRB-", "-RRB-", ",", ":", ".", "''", '""', "``", "#", "$", 
    "ADD", "AFX", "BES", "CC", "CD", "DT", "EX", "FW", "GW", "HVS", 
    "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NFP", "NIL", "NN", 
    "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", 
    "RBS", "RP", "_SP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", 
    "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "XX"
]
ent_types = [
    "", "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", 
    "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", 
    "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"

]
ent_iobs = [
    "I", "O", "B", ""
]

tag_num = len(tags)
ent_type_num = len(ent_types)
ent_iob_num = len(ent_iobs)
other_features_num = 21
batch_size = 50


def get_sent_features(sent, sent_id, tfidf_doc):
    noun_chunks = list(sent.noun_chunks)
    sent_len = len(list(sent))
    # tokenized_sent = [token.lemma_ for token in sent if not (token.is_punct or token.lemma_ in stopwords_set)]
    # data = get_word_ids([tokenized_sent], vocab, max_length=75)
    # cnn_output = cnn_model.predict([data])[0][0]

    if sent_id < 5:
        sent_id = "<5"
    elif sent_id < 10:
        sent_id = "<10"
    elif sent_id < 20:
        sent_id = "<20"
    elif sent_id < 50:
        sent_id = "<50"
    else:
        sent_id = ">=50"

    if sent_len < 5:
        sent_len = "<5"
    elif sent_len < 10:
        sent_len = "<10"
    elif sent_len < 20:
        sent_len = "<20"
    elif sent_len < 50:
        sent_len = "<50"
    else:
        sent_len = ">=50"

    features = {
        # offset and length info
        "sent_id": sent_id, 
        "sent_len": sent_len, 


        # tfidf info
        "root_tfidf": tfidf_doc[sent.root.lemma_], 
        "avg_tfidf": sum([tfidf_doc[token.lemma_] for token in sent if not token.is_punct and token.lemma_ in tfidf_doc]) / (sent.end - sent.start), 
        "max_tfidf": max([calc_nc_avg_tfidf(nc, tfidf_doc) for nc in noun_chunks]) if noun_chunks else 0., 

        # orth info
        "has_THE-R": any(["THE-R" in token.text for token in sent]), 
        "has_quote": '"' in sent.text, 
        "has_patent": True if re.findall(r"[Pp]atent", sent.text) else False,   
        "has_num_patent": True if re.findall(r"[0-9]+ [Pp]atent", sent.text) else False,   
        "As": sent.text.startswith("As"), 
        "describe": any([token.lemma_ == "describe" for token in sent]), 
        "explain": any([token.lemma_ == "explain" for token in sent]), 
        "claim": any([token.lemma_ == "claim" for token in sent]), 
        "method": any([token.lemma_ == "method" for token in sent]), 
        "direct": any([token.lemma_ == "direct" for token in sent]), 
        "entitle": any([token.lemma_ == "entitle" for token in sent]), 
        "relates to": any([token.lemma_ == "relate" for token in sent]), 
        "disclose": any([token.lemma_ == "disclose" for token in sent]), 
        "is the assignee": any([token.lemma_ == "assign" for token in sent]), 
        "own": any([token.lemma_ == "own" for token in sent]), 
        "comprising": any([token.lemma_ == "comprise" for token in sent]), 
        "involve": any([token.lemma_ == "involve" for token in sent]), 
        "concern": any([token.lemma_ == "concern" for token in sent]), 
        "discover": any([token.lemma_ == "discover" for token in sent]), 
        "recite": any([token.lemma_ == "recite" for token in sent]), 
        
        # # cnn_output
        # "cnn_output": cnn_output
    }

    return features


def load_file_data_label(file_id):
    with open(os.path.join(spacy_docs_dir, spacy_docs_list[file_id]), "rb") as f:
        doc = pickle.load(f)
    tfidf_doc = tfidfs[file_id]
    sent_labels_doc = load_sent_labels(file_id)
    # print(sent_labels_doc)

    sent_num = sum([1 for t in doc.sents])
    for idx, sent in enumerate(doc.sents):
        if idx >= 0.4 * sent_num:
            break
        if filter_sent(sent):
            data = get_sent_features(sent, idx, tfidf_doc)
            label = sent_labels_doc[idx][1]
            # label = random.random() < 0.5 # randomly generate labels
            yield idx, data, label
            if label == 1.:
                for _ in range(60):
                    yield idx, data, label


def load_dataset(file_ids):
    for i in file_ids:
        file_name = spacy_docs_list[i]
        print(file_name)
        yield from load_file_data_label(i)

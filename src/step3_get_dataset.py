import pickle
import numpy as np
import re
import string
import os
from utils import load_stuff,\
    spacy_docs_dir, spacy_docs_list, \
    filter_noun_chunks, \
    load_pt_answers, stopwords_set, \
    get_normalized_noun_chunk

tfidfs = load_stuff("tfidfs")
pt_answers = load_pt_answers()


try:
    org_and_per_names = load_stuff("org_and_per_names")
except FileNotFoundError:
    print("FileNotFoundError: org_and_per_names.pk not found!")
    pass
person_titles = ["Dr.", "Sir", "Mr.", "Mrs.", "Judge"]


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


def get_features(nc, doc, tfidf_doc, first_appearance=None):
    root = nc.root
    head = root.head

    features = {
        # offset and length info
        "doc_offset": first_appearance if first_appearance else nc.start, # first appearance
        # "sent_offset": nc.start - nc.sent.start, 
        # "token_num": nc.end - nc.start, 
        # "char_len": nc.end_char - nc.start_char, 
        # "sent_len": len(nc.sent), 

        # tfidf info
        "root_tfidf": tfidf_doc[nc.root.lemma_], 
        "avg_tfidf": sum([tfidf_doc[token.lemma_] for token in nc if not token.is_punct and token.lemma_ in tfidf_doc]) / (nc.end - nc.start), 
        
        # # pos tag info
        "root_pos_tag": root.tag_, 
        # "pos_tags": tuple([token.tag_ for token in nc]), 
        "head_pos_tag": head.tag_, 

        # # dependency info
        "root_dep": root.dep_, 
        "head_dep": head.dep_, 

        # # orth info
        "has_THE-R": any(["THE-R" in token.text for token in nc]), 
        "beside_THE-R": any([
            "THE-R" in text for text in
            [
                doc[nc.start - 1].text if nc.start >= 1 else "", 
                doc[nc.start - 2].text if nc.start >= 2 else "", 
                doc[nc.end + 1].text if nc.end + 1 < len(doc)  else "", 
                doc[nc.end + 2].text if nc.end + 2 < len(doc) else "", 
            ]
            ]), 

        # # orth info
        "root_shape": root.shape_, 
        "has_hyphen": any(['-' in token.text for token in nc]), 
        "root_has_hyphen": '-' in root.text, 
        "has_quote": any(['"' in token.text or "'" for token in nc]), 
        "root_has_num": '"' in root.text or "'" in root.text, 

        "has_capitalized": any([t.text[0] in string.ascii_uppercase for t in nc]), 
        "is_all_lower": nc.text.lower() == nc.text, 
        "has_digit": any([c.isdigit() for c in nc.text]), 
        "has_capital": any([c in string.ascii_uppercase for c in nc.text])
    }

    return features


def get_label(nc, gold=None):
    if any([pt in nc.text for pt in gold]):
        return 1.
    return 0.
    # return 0. if np.random.random_sample() < 0.5 else 1.


def load_file_data_label(file_id):
    with open(os.path.join(spacy_docs_dir, spacy_docs_list[file_id]), "rb") as f:
        doc = pickle.load(f)
    tfidf_doc = tfidfs[file_id]

    nc_num = sum([1 for t in doc.noun_chunks])
    nc_dict = {} # record the first appearance position of each noun chunk

    for idx, nc in enumerate(doc.noun_chunks):
        if idx >= 0.4 * nc_num:
            break
        nc_text = get_normalized_noun_chunk(nc, file_id)
        # if filter_noun_chunks(nc, file_id):
            # if nc_text not in nc_dict:
            #     data = get_features(nc, doc, tfidf_doc)
            #     nc_dict[nc_text] = data["doc_offset"]
            # else:
            #     data = get_features(nc, doc, tfidf_doc, nc_dict[nc_text])
        if filter_noun_chunks(nc, file_id) and nc_text not in nc_dict:
            data = get_features(nc, doc, tfidf_doc)
            
            label = get_label(nc, gold=pt_answers[file_id])
            yield nc, data, label
            if label == 1.:
                for _ in range(20):
                    yield nc, data, label


def load_dataset(file_ids):
    for i in file_ids:
    # for i in range(2):
        file_name = spacy_docs_list[i]
        print(file_name)
        yield from load_file_data_label(i)
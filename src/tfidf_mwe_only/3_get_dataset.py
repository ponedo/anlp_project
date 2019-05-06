import pickle
import numpy as np
import re
import string
from utils import *


w2v = load_word_embeddings()
tfidfs = load_stuff("tfidfs.pk")

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

word_emb_dim = word_emb_dim
tag_num = len(tags)
ent_type_num = len(ent_types)
ent_iob_num = len(ent_iobs)
other_features_num = 21
batch_size = 50


def read_sents(file_id):
    file_name = pos_ner_tagged_data_file_list[file_id]
    print(file_name)
    with open(os.path.join(pos_ner_tagged_data_dir, file_name), "rb") as f:
        return pickle.load(f)


def get_feature(nc, doc, tfidf_doc):
    start, end = nc.start, nc.end
    text, tag_, ent_type_, ent_iob_ = nc.text, nc.tag_, nc.ent_type_, nc.ent_iob_
    
    if text.lower() in w2v:
        word_emb = w2v[text.lower()]
    else:
        word_emb = np.zeros(word_emb_dim)
    
    tag = tags.index(tag_)
    tag_emb = np.zeros(tag_num, dtype=np.float32)
    tag_emb[tag] = 1.0

    ent_type = ent_types.index(ent_type_)
    ent_type_emb = np.zeros(ent_type_num, dtype=np.float32)
    ent_type_emb[ent_type] = 1.0

    ent_iob = ent_iobs.index(ent_iob_)
    ent_iob_emb = np.zeros(ent_iob_num, dtype=np.float32)
    ent_iob_emb[ent_iob] = 1.0

    is_capitalized = text[0] in string.ascii_uppercase
    is_all_caps = text.upper() == text
    is_all_lower = text.lower() == text
    is_numeric = text.isdigit()
    capitals_inside = text[1:].lower() != text[1:]
    has_num = any([num in text for num in string.digits])
    has_underscore = "_" in text
    has_hyphen = '-' in text
    court = "court" in text.lower()
    patent = "patent" in text.lower()
    company = any([c in text.lower() for c in ["ltd.", "inc.", "co." ,"corp."]])
    the_r = "THE_R" in text
    law = any([l in text for l in ["col.", ""]])
    patent_num = re.search("(?:[\′\'][0-9]+\s[Pp]atent)", text) # ′411 patent
    length_1 = len(text) == 1
    length_2 = len(text) == 2
    length_3 = len(text) == 3
    length_4_6 = len(text) >= 4 and len(text) <= 6
    length_7_10 = len(text) >= 7 and len(text) <= 10
    length_11_up = len(text) >= 11
    tfidf_value = tfidf[text] if text in tfidf else 0.0

    other_features = [
        is_capitalized, is_all_caps, is_all_lower, is_numeric, 
        capitals_inside, has_num, has_underscore, has_hyphen, 
        court, patent, company, the_r, law, patent_num, 
        length_1, length_2, length_3, length_4_6, length_7_10, 
        length_11_up, tfidf_value
    ]
    for i in range(len(other_features)):
        other_features[i] = 1.0 if other_features[i] else 0.0
    other_features = np.array(other_features)

    return np.concatenate(
        (word_emb, tag_emb, ent_type_emb, ent_iob_emb, 
        other_features))


def pipeline(file_id):
    sents = read_sents(file_id)
    tfidf = tfidfs[file_id]

    sents_data_label = []
    for sent in sents:
        sent_data = [get_feature(token, tfidf) for token in sent[0]]
        sent_label = sent[1]
        sents_data_label.append((sent_data, sent_label))

    return sents_data_label


if __name__ == "__main__":
    """
        Handle the data & labels into format like following:
        [sent_data_label
            [sent
                [sent_data
                    [token_feature_array]
                    [token_feature_array]
                    [token_feature_array]
                    ...
                ]
                [sent_label
                    [token_label]
                    [token_label]
                    [token_label]
                    ...
                ]
            ]
            ...
        ]
    """
    batch_num = 0
    sents_data_label_batch = []
    for i in range(len(pos_ner_tagged_data_file_list)):
        sents_data_label_doc = pipeline(i)
        sents_data_label_batch.extend(sents_data_label_doc)
        while len(sents_data_label_batch) >= batch_size:
            save_stuff(
                sents_data_label_batch[:batch_size], 
                os.path.join(data_label_batches_dir, "sents_data_label_") + str(batch_num))
            batch_num += 1
            sents_data_label_batch = sents_data_label_batch[batch_size:]
        save_stuff(
            sents_data_label_doc, 
            os.path.join(data_label_docs_dir, "sents_data_label_" + os.listdir(pos_ner_tagged_text_dir)[i]))
    save_stuff(
        sents_data_label_batch, 
        os.path.join(data_label_batches_dir, "sents_data_label_") + str(batch_num))
    
import os
import re
import pickle
import gensim
from nltk.corpus import stopwords

word_emb_dim = 64

raw_data_dir = "..\\ivan_data\\merged"
regular_results_dir = "..\\ivan_data\\regular_results"
preprocessed_tokens_dir = "..\\ivan_data\\preprocessed\\tokens"
preprocessed_sents_dir = "..\\ivan_data\\preprocessed\\sents"
tfidf_results_dir = "..\\ivan_data\\tfidf_results"
spacy_docs_dir = "..\\ivan_data\\spacy_docs"
patent_terms_tfidf_only_dir = "..\\ivan_data\\patent_terms\\tfidf_only"
patent_terms_log_reg_dir = "..\\ivan_data\\patent_terms\\log_reg"
key_sents_tfidf_only_dir = "..\\ivan_data\\key_sents\\tfidf_only"
key_sents_log_reg_dir = "..\\ivan_data\\key_sents\\log_reg"

raw_file_list = os.listdir(raw_data_dir)
preprocessed_tokens_file_list = os.listdir(preprocessed_tokens_dir)
preprocessed_sents_file_list = os.listdir(preprocessed_sents_dir)
tfidf_results_list = os.listdir(tfidf_results_dir)
spacy_docs_list = os.listdir(spacy_docs_dir)
patent_terms_tfidf_only_list = os.listdir(patent_terms_tfidf_only_dir)
patent_terms_log_reg_list = os.listdir(patent_terms_log_reg_dir)
key_sents_tfidf_only_list = os.listdir(key_sents_tfidf_only_dir)
key_sents_log_reg_list = os.listdir(key_sents_log_reg_dir)


word2vec_dir = "..\\models"
lstm_dir = "..\\models\\lstm"
answer_tsv_path = "../ivan_data/answer.tsv"
useless_terms_path = "../ivan_data/useless_terms.txt"
useless_noun_chunks_path = "../ivan_data/useless_noun_chunks.txt"


#####################
# File IO functions #
#####################
def save_stuff(stuff, file_name):
    with open(file_name + ".pk", "wb") as f:
        pickle.dump(stuff, f)


def load_stuff(file_name):
    with open(file_name + ".pk", "rb") as f:
        return pickle.load(f)


def load_useless_terms():
    useless_terms = []
    with open(useless_terms_path, "r", encoding="cp950") as f:
        for line in f:
            line = line.strip()
            useless_terms.append(line)
    return useless_terms
useless_terms = load_useless_terms()


def load_useless_noun_chunks():
    useless_noun_chunks = []
    with open(useless_noun_chunks_path, "r", encoding="cp950") as f:
        for line in f:
            line = line.strip()
            useless_noun_chunks.append(line)
    return useless_noun_chunks
useless_noun_chunks = load_useless_noun_chunks()


def load_word_embeddings():
    return gensim.models.KeyedVectors.load_word2vec_format(r"../models/minc3_nega3_sg0_hs1.txt")


def load_pt_answers():
    answers = []
    print("loading answers.tsv...")
    with open(answer_tsv_path, "r", encoding="utf-8") as f:
        firstline = True
        i = 0
        for line in f:
            if firstline:
                firstline = False
                continue
            i += 1
            splitted_line = line.strip().split("\t")
            file_name = splitted_line[0]
            patent_terms = splitted_line[1]
            sents = splitted_line[2]
            patent_terms = patent_terms.strip().split(";")
            answers.append(patent_terms)
    return answers


def load_sent_labels(file_id):
    sent_labels_doc = []
    with open(os.path.join(preprocessed_sents_dir, preprocessed_sents_file_list[file_id]), "r", encoding="cp950") as f:
        for line in f:
            line = line.strip("\n ").split("\t")
            if len(line) >= 2:
                sent_text = line[1]
                if not line[0]:
                    sent_labels_doc.append((sent_text, 0.))
                else:
                    sent_labels_doc.append((sent_text, 1.))
    return sent_labels_doc


#####################################################
# Token / Noun chunks / Sentences handler functions #
#####################################################
try:
    org_and_per_names = load_stuff("org_and_per_names")
except FileNotFoundError:
    print("FileNotFoundError: org_and_per_names.pk not found!")
    pass
person_titles = ["Dr.", "Sir", "Mr.", "Mrs.", "Judge"]


def get_stopwords_set():
    stopwords_set = set(stopwords.words('english'))
    for word in ["a", "an", "the", "this", "that", "these", "those", "An", "This", "That", "These", "Those", "\n"]:
            stopwords_set.add(word)
    return stopwords_set
stopwords_set = get_stopwords_set()


def filter_token(t, file_id):
    return not (
        t.lemma_ in stopwords_set \
        or t.is_punct \
        or all([not c.isalpha() for c in t.lemma_]) \
        or t.text in org_and_per_names[file_id] \
        or t.text in person_titles \
        or t.text == "'s")


def filter_noun_chunks(nc, file_id):
    # print(nc.root.dep_)
    # if nc.dep_ == "nsubj":
    #     return False
    nc_text = get_normalized_noun_chunk(nc, file_id)
    if nc_text in useless_noun_chunks:
        return False

    if "said" in nc.text \
        or "col." in nc.text \
        or "i.e." in nc.text \
        or "e.g." in nc.text:
        # or "'s" in nc.text \
        return False
    if sum([c.isdigit() for c in nc.text]) >= 2:
        return False

    org_and_per_names_doc = org_and_per_names[file_id]
    if any([token.lemma_ in org_and_per_names_doc for token in nc]) \
        and "'s" not in nc.text:
        return False
    if any([token.lemma_ in person_titles for token in nc]) \
        and "'s" not in nc.text:
        return False
    if all([not c.isalpha() for c in nc.text]):
        return False
    
    
    if any([token.text in stopwords_set for token in nc]) \
        and len(nc) <= 2 and all([token.lemma_.lower() == token.lemma_ for token in nc]):
        return False

    # if re.match(r"′[0-9]+ [Pp]atent", nc.text) \
    #     or re.match(r"' [0-9]+ [Pp]atent", nc.text):
    if re.match(r".*[0-9]+ Pp]atent.*", nc.text):
        print(nc.text)
        return False
        
    return True


def filter_sent(sent):
    if len(sent.text.strip().split()) == 1:
        return False
    
    if all([not c.isalpha() for c in sent.text]):
        return False

    if (any([abbr in sent.text for abbr in ["Col.", "col.", "Id.", "ll.", "l."]]) or re.findall(r"[0-9\-]+", sent.text))\
        and (len(sent) <= 4):
        return False
    
    return True


def get_normalized_noun_chunk(nc, file_id):
    normalized =  " ".join([t.lemma_ for t in nc if filter_token(t, file_id)])
    for useless_term in useless_terms:
        pattern = r"(?:\s" + useless_term + r"\s)" \
            + r"|" + r"(?:^" + useless_term + r"\s)" \
            + r"|" + r"(?:\s" + useless_term + r"$)" \
            + r"|" + r"(?:^" + useless_term + r"$)"
        pre_normalized = normalized
        normalized = re.sub(pattern, " ", normalized)
        while pre_normalized != normalized:
            pre_normalized = normalized
            normalized = re.sub(pattern, " ", normalized)
        normalized = normalized.strip()
    return normalized
        

def get_normalized_sent(sent, file_id):
    return None


def calc_nc_avg_tfidf(nc, tfidf_doc):
    tfidf_each_token = [tfidf_doc[token.lemma_] for token in nc if token.lemma_ in tfidf_doc]
    try:
        avg_tfidf = sum(tfidf_each_token) / len(tfidf_each_token)
    except ZeroDivisionError:
        print("To be filtered noun chunk: ", nc.text)
    return avg_tfidf
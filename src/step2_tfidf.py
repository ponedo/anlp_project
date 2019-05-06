### Some code is from class materials ###

import spacy, glob, os, operator, math, random
import pickle
from collections import Counter
from utils import filter_noun_chunks, filter_token, \
    spacy_docs_dir, spacy_docs_list, \
    preprocessed_tokens_dir, answer_tsv_path, \
    patent_terms_tfidf_only_dir, key_sents_tfidf_only_dir, \
    load_stuff, save_stuff, load_pt_answers, load_sent_labels, \
    stopwords_set, get_normalized_noun_chunk, calc_nc_avg_tfidf
from collections import Counter


# load the answers
pt_answers = load_pt_answers()
pt_ranks = []
pt_predictions = []
pt_unpredicted = {}
sent_ranks = []
sent_predictions = []
sent_unpredicted = {}
deps = Counter()

try:
    org_and_per_names = load_stuff("org_and_per_names")
except FileNotFoundError:
    print("FileNotFoundError: org_and_per_names.pk not found!")
    pass
person_titles = ["Dr.", "Sir", "Mr.", "Mrs.", "Judge"]


def read_docs(inputDir):
    """ Read in documents (all ending in .txt) from an input folder"""
    docs=[]
    for idx, filename in enumerate(glob.glob(os.path.join(inputDir, '*.txt'))):
        with open(filename, "r", encoding="cp950") as f:
            tokens = []
            for line in f:
                line = line.strip()
                if line:
                    tokens.append(line.split("\t")[0])

            docs.append((idx, tokens))
    return docs


def tf_idf_ranking(docs):
    """
    Function to rank terms in document by tf-idf score, and return the top 10 terms
    
    Input: a list of (filename, [spacy tokens]) documents
    Returns: a dict mapping "filename" -> [list of 10 keyphrases, ranked from highest tf-idf score to lowest]
    """
    def get_tf(tokens):
        counter=Counter()
        for token in tokens:
            # remove something
            counter[token]+=1
        token_num = len(counter)
        for token in counter:
            counter[token] /= token_num
            counter[token] = math.log(counter[token] + 1)
        return counter
    
    def get_idfs(docs):
        counts=Counter()
        for _, doc in docs:
            doc_types={}
            for token in doc:
                doc_types[token]=1

            for word in doc_types:
                counts[word]+=1

        idfs={}
        for term in counts:
            idfs[term]= math.pow(math.log(float(len(docs))/counts[term]), 4.0)

        return idfs

    # print("getting_idf")
    idfs=get_idfs(docs)
    # print("idf_ok")

    keyphrases={}
    tfidfs = {}
    
    for filename, doc in docs:
        # print("getting_tf", filename)
        tf=get_tf(doc)
        candidates={}
        for term in tf:
            candidates[term]=tf[term]*idfs[term]

        sorted_x = sorted(candidates.items(), key=operator.itemgetter(1), reverse=True)
       
        keyphrases[filename]=[k for k,v in sorted_x]
        tfidfs[filename] = candidates
        # keyphrases[filename]=[k for k,v in sorted_x[:10]]
    # print("tf_ok")
    
    return keyphrases, tfidfs


def get_tfidf(docs_dir, merged_mwe=True):
    if merged_mwe:
        docs = read_docs(docs_dir)
    else:
        docs = read_docs(docs_dir)
    terms, tfidfs = tf_idf_ranking(docs)
    if merged_mwe:
        save_stuff(terms, "keyphrases_mwe")
        save_stuff(tfidfs, "tfidfs_mwe")
    else:
        save_stuff(terms, "keyphrases")
        save_stuff(tfidfs, "tfidfs")
    return tfidfs, terms


def extract_patent_terms(doc, file_id):
    file_name = spacy_docs_list[file_id]
    tfidf_doc = tfidfs[file_id]

    # get the noun_chunks in the doc, with their tfidf values and whether they are patent terms
    nc_num = sum([1 for _ in doc.noun_chunks])

    # iterate over first 40% noun chunks
    nc_tfidf_dict = {}
    for idx, nc in enumerate(doc.noun_chunks):
        if idx >= 0.4 * nc_num:
            break
        # normalize the shape of noun chunks
        # e.g.: both "great things" and "greater thing" are normalized into "great thing"
        # which eliminates nonsense duplicate
        # Also, eliminate names of organization and persons here
        nc_text = get_normalized_noun_chunk(nc, file_id)

        # filter the noun chunks and only those needed remain
        if filter_noun_chunks(nc, file_id) and nc_text not in nc_tfidf_dict:
            ################################## FILTER AT BELOW LINE? #######################
            # tfidf_each_token = [tfidf_doc[token.lemma_] for token in nc if token.lemma_ in tfidf_doc and filter_token(token, file_id)]
            avg_tfidf = calc_nc_avg_tfidf(nc, tfidf_doc)

            if any([pt in nc.text for pt in pt_answers[file_id]]):
                nc_tfidf_dict[nc_text] = (avg_tfidf, True, nc.root.dep_)
            
            else:
                nc_tfidf_dict[nc_text] = (avg_tfidf, False, nc.root.dep_)

    # merge those noun chunks that are substrings of another noun chunks
    # e.g. with both "pill" and "pill A" and "other pills" extracted, single token "pill" is removed
    to_merge = []
    for nc_text0 in nc_tfidf_dict:
        for nc_text1 in nc_tfidf_dict:
            if nc_text0 not in to_merge and nc_text0 != nc_text1 and nc_text0 in nc_text1:
                to_merge.append(nc_text0)
    for nc_text in to_merge:
        del(nc_tfidf_dict[nc_text])

    # sort the noun chunks with tfidf ranking
    tfidf_only_results = sorted(nc_tfidf_dict.items(), key=lambda x: x[1][0], reverse=True)

    # record the highest ranking answer
    pt_found = False
    with open(os.path.join(patent_terms_tfidf_only_dir, file_name), "w", encoding="cp950") as f:
        for idx, (nc_text, (tfidf, pt, dep_)) in enumerate(tfidf_only_results):
            if pt:
                deps[dep_] += 1
                f.write("※")
                if not pt_found:
                    pt_ranks.append(idx + 1)
                    pt_predictions.append(nc_text)
                    pt_found = True
            f.write("\t" + nc_text + "\t" + str(tfidf) + "\n")
    if not pt_found:
        pt_unpredicted[file_id] = file_name
        pt_ranks.append("Not found")


def extract_key_sents(doc, file_id):
    file_name = spacy_docs_list[file_id]
    tfidf_doc = tfidfs[file_id]
    sent_labels_doc = load_sent_labels(file_id)
    answer_ids = [i for i in range(len(sent_labels_doc)) if sent_labels_doc[i][1] != 0.]
    
    sent_tfidf = {}
    # get the average term-level tfidf of the each sentence
    for idx, sent in enumerate(doc.sents):
        acc_tfidf, token_num = 0., 0
        for token in sent:
            #################### filter tokens (stopwords?) ################ 
            if filter_token(token, file_id) and token.lemma_ in tfidf_doc:
                acc_tfidf += tfidf_doc[token.lemma_]
                token_num += 1
        try:
            avg_tfidf = acc_tfidf / token_num
        except ZeroDivisionError:
            avg_tfidf = 0.
        sent_tfidf[idx] = avg_tfidf
    
    # sort the sents by avg_tfidf
    sent_tfidf = sorted(sent_tfidf.items(), key=lambda x: x[1], reverse=True)

    # write the result into file
    sent_found = False
    with open(os.path.join(key_sents_tfidf_only_dir, file_name), "w", encoding="cp950") as f:
        for idx, (sent_id, avg_tfidf) in enumerate(sent_tfidf):
            sent_text = sent_labels_doc[sent_id][0]
            if sent_id in answer_ids:
                f.write("※")
                if not sent_found:
                    sent_ranks.append(idx + 1)
                    sent_predictions.append(sent_text)
                    sent_found = True
            f.write("\t" + sent_text + "\t" + str(avg_tfidf) + "\n")
    if not sent_found:
        sent_unpredicted[file_id] = file_name
        sent_ranks.append("Not found")


def pipeline(file_id, top_term=None):
    file_name = spacy_docs_list[file_id]
    print(file_name)

    # read the doc
    with open(os.path.join(spacy_docs_dir,file_name), "rb") as f:
        doc = pickle.load(f)

    extract_patent_terms(doc, file_id)
    extract_key_sents(doc, file_id)


if __name__ == "__main__":
    # try:
    #     tfidfs = load_stuff("tfidfs")
    #     print("\'tfidf.pk\' found.")
    # except FileNotFoundError:
    #     print("FileNotFoundError: \'tfidf.pk\' doesn't exist. Reproducing tfidf using the preprocessed documents")
    #     tfidfs, terms = get_tfidf(preprocessed_tokens_dir, False)

    tfidfs, terms = get_tfidf(preprocessed_tokens_dir, False)

    # pipeline(101, tfidfs[101])

    for i in range(len(spacy_docs_list)):
    # for i in range(100, 140):
        pipeline(i)

    # print(predictions)
    only_int_pt_ranks = [rank for rank in pt_ranks if isinstance(rank, int)]
    print("====================================== Patent terms results =============================================")
    print("Patent term ranks: ", pt_ranks)
    print("Patent term unpredicted: ", pt_unpredicted)
    print("Patent term predicted: %d" % len(only_int_pt_ranks))
    print("Number of documents whose patent terms extracted ok (answers within top 5 rank): %d" % (len([r for r in only_int_pt_ranks if r <= 5])))
    print("Patent term average rank: %f" % (sum(only_int_pt_ranks) / len(only_int_pt_ranks)))
    only_int_sent_ranks = [rank for rank in sent_ranks if isinstance(rank, int)]
    print("====================================== Key sentences results =============================================")
    print("Key sentences ranks: ", sent_ranks)
    print("Key sentences unpredicted: ", sent_unpredicted)
    print("Key sentences predicted: %d" % len(only_int_sent_ranks))
    print("Number of documents whose key sentences extracted ok (answers within top 5 rank): %d" % (len([r for r in only_int_sent_ranks if r <= 5])))
    print("Key sentences average rank: %f" % (sum(only_int_sent_ranks) / len(only_int_sent_ranks)))
    
import spacy, glob, os, operator, math, random
import pickle
from utils import *
from collections import Counter


nlp = spacy.load('en')


def read_docs(inputDir):
    """ Read in movie documents (all ending in .txt) from an input folder"""
    
    docs=[]
    for idx, filename in enumerate(glob.glob(os.path.join(inputDir, '*.txt'))):
        with open(filename, "rb") as file:
            docs.append((idx, pickle.load(file)))
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
            counter[token.text]+=1
        return counter
    
    def get_idfs(docs):
        counts=Counter()
        for _, doc in docs:
            doc_types={}
            for token in doc:
                doc_types[token.text]=1

            for word in doc_types:
                counts[word]+=1

        idfs={}
        for term in counts:
            idfs[term]=math.log(float(len(docs))/counts[term])

        return idfs

    idfs=get_idfs(docs)

    keyphrases={}
    tfidfs = {}
    
    for filename, doc in docs:
        tf=get_tf(doc)
        candidates={}
        for term in tf:
            candidates[term]=tf[term]*idfs[term]

        sorted_x = sorted(candidates.items(), key=operator.itemgetter(1), reverse=True)
       
        keyphrases[filename]=[k for k,v in sorted_x]
        tfidfs[filename] = candidates
        # keyphrases[filename]=[k for k,v in sorted_x[:10]]
    
    return keyphrases, tfidfs


def keyphrase_no_some(docs):
    """
    Function to rank terms in document by tf-idf score, and return the top 10 terms.  
    Constraint: None of the top 10 terms should be proper nouns.
    
    Input: a list of (filename, [spacy tokens]) documents
    Returns: a dict mapping "filename" -> [list of 10 keyphrases, ranked from highest tf-idf score to lowest]
    
    """
    
    def remove_some(docs):
        new_docs=[]
        for filename, doc in docs:
            new_doc=[]
            for token in doc:
                if token.ent_type_ != "ORG" and token.ent_type_ != "ORG":
                    new_doc.append(token)
            new_docs.append((filename, new_doc))
       
        return new_docs
            
    new_docs = remove_some(docs)
    terms, tfidfs = tf_idf_ranking(new_docs)
    return terms, tfidfs


def get_tfidf(docs_dir, merged_mwe=True):
    if merged_mwe:
        docs = read_docs(preprocessed_data_dir)
    else:
        docs = read_docs(raw_data_dir)
    terms, tfidfs = tf_idf_ranking(docs)
    if merged_mwe:
        save_stuff(terms, "keyphrases_mwe")
        save_stuff(tfidfs, "tfidfs_mwe")
    else:
        save_stuff(terms, "keyphrases")
        save_stuff(tfidfs, "tfidfs")
    return tfidfs, terms


def pipeline(file_id, tfidfs, terms, top_term=10, top_sent=5):
    preprocessed_text = read_preprocessed_file(file_id)
    file_name = preprocessed_file_list[file_id]

    tfidfs_file = tfidfs[file_id]
    terms_file = terms[file_id]

    print(file_name)
    f = open(os.path.join(tfidf_results_dir, file_name), "w", encoding="cp950")

    f.write("Keyphrases:\n\n")
    for term in terms_file[:top_term]:
        f.write(term)
        f.write("\n")

    f.write("====================================\n")
    f.write("top sentences:\n\n")

    sents = [s for s in preprocessed_text.strip().split("\n") if s]
    sents_with_tfidf = {}
    for idx, sent in enumerate(sents):
        tokens = [token for token in sent.strip().split()]
        tfidf_sent, total = 0, 0
        for token in tokens:
            if token in tfidfs_file:
                total += 1
                tfidf_sent += tfidfs_file[token]
        if total == 0:
            continue
        tfidf_sent /= total
        sents_with_tfidf[idx] = tfidf_sent
    
    sents_with_tfidf = sorted(sents_with_tfidf.items(), key=lambda x: x[1], reverse=True)
    top_sents = [(sents[idx], ti) for idx, ti in sents_with_tfidf[:top_sent]]
    for top_sent, tfidf in top_sents:
        f.write(top_sent)
        f.write("\n%f\n" % tfidf)

    f.close()


if __name__ == "__main__":
    try:
        tfidfs = load_stuff("tfidfs_mwe")
        terms = load_stuff("keyphrases_mwe")
    except FileNotFoundError:
        tfidfs, terms = get_tfidf(preprocessed_data_dir, True)

    for i in range(len(preprocessed_file_list)):
        pipeline(i, tfidfs, terms)
    
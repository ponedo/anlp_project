import os
from utils import load_stuff, \
    patent_terms_log_reg_dir, spacy_docs_list, \
    load_pt_answers, filter_noun_chunks, \
    stopwords_set, get_normalized_noun_chunk
from step3_get_dataset import load_file_data_label
from collections import Counter


pipeline = load_stuff("clf_pt")
pt_answers = load_pt_answers()
ranks = []
predictions = []
unpredicted = {}
deps = Counter()

try:
    org_and_per_names = load_stuff("org_and_per_names")
except FileNotFoundError:
    print("FileNotFoundError: org_and_per_names.pk not found!")
    pass
person_titles = ["Dr.", "Sir", "Mr.", "Mrs.", "Judge"]


def extract_patent_terms(file_id):
    file_name = spacy_docs_list[file_id]
    print(file_name)

    noun_chunks, data, _ = zip(*load_file_data_label(file_id))
    y_prob_ = pipeline.predict_proba(data)
    
    result_dict = {}
    for nc, _, prob in zip(noun_chunks, data, y_prob_):
        nc_text = get_normalized_noun_chunk(nc, file_id)
        if filter_noun_chunks(nc, file_id) and nc_text not in result_dict:
            if any([pt in nc.text for pt in pt_answers[file_id]]):
                result_dict[nc_text] = ([], True, nc.root.dep_)
            else:
                result_dict[nc_text] = ([], False, nc.root.dep_)
        result_dict[nc_text][0].append(prob[1])

    results = {}
    for nc_text, (v, pt, dep_) in result_dict.items():
        avg_proba = sum(v) / len(v)
        results[nc_text] = (avg_proba, pt, dep_)
    results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
    
    pt_found = False
    with open(os.path.join(patent_terms_log_reg_dir, file_name), "w", encoding="cp950") as f:
        for idx, (nc_text, (proba, pt, dep_)) in enumerate(results):
            if pt:
                deps[dep_] += 1
                f.write("※")
                # if "ubj" not in dep_ and "obj" not in dep_:
                #     print(nc_text, dep_)
                if not pt_found:
                    ranks.append(idx + 1)
                    predictions.append(nc_text)
                    # print(idx+1)
                    pt_found = True
            f.write("\t" + nc_text + "\t" + str(proba) + "\n")
    if not pt_found:
        unpredicted[file_id] = file_name
        ranks.append("Not found")

    return results


if __name__ == "__main__":
    # for i in range(len(spacy_docs_list)):
    for i in range(100, 140):
        extract_patent_terms(i)

    only_int_ranks = [rank for rank in ranks if isinstance(rank, int)]
    print({k: v / sum(deps.values()) for k, v in deps.items()})
    # print(predictions)
    print("====================================== Patent terms results =============================================")
    print("Patent term ranks: ", ranks)
    print("Patent term unpredicted: ", unpredicted)
    print("Patent term predicted: %d" % len(only_int_ranks))
    print("Number of documents whose patent terms extracted ok (answers within top 5 rank): %d" % (len([r for r in only_int_ranks if r <= 5])))
    print("Patent term average rank: %f" % (sum(only_int_ranks) / len(only_int_ranks)))
    
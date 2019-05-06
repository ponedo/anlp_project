import os
from utils import load_stuff, \
    key_sents_log_reg_dir, spacy_docs_list, \
    load_sent_labels, filter_sent
from step6_get_sent_dataset import load_data_label_for_CNN
from step7_train_CNN import get_word_ids, vocab


cnn_model = load_stuff("CNN")
cnn_model.load_weights("CNN.hdf5")
ranks = []
predictions = []
unpredicted = {}


def extract_key_sents(file_id):
    file_name = spacy_docs_list[file_id]
    print(file_name)

    sent_labels_doc = load_sent_labels(file_id)
    answer_ids = [i for i in range(len(sent_labels_doc)) if sent_labels_doc[i][1] != 0.]
    
    sents, data, _ = zip(*load_data_label_for_CNN(file_id))
    data = get_word_ids(data, vocab, max_length=75)
    y_prob_ = cnn_model.predict(data)

    result_dict = {}
    for sent_id, _, prob in zip(sents, data, y_prob_):
        result_dict[sent_id] = prob[0]

    results = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)

    # write the result into file
    sent_found = False
    with open(os.path.join(key_sents_log_reg_dir, file_name), "w", encoding="cp950") as f:
        for idx, (sent_id, prob) in enumerate(results):
            sent_text = sent_labels_doc[sent_id][0]
            if sent_id in answer_ids:
                f.write("※")
                if not sent_found:
                    ranks.append(idx + 1)
                    predictions.append(sent_text)
                    # print(idx+1)
                    sent_found = True
            f.write("\t" + sent_text + "\t" + str(prob) + "\n")
    if not sent_found:
        unpredicted[file_id] = file_name
        ranks.append("Not found")

    return results


if __name__ == "__main__":
    # for i in range(len(spacy_docs_list)):
    for i in range(100, 140):
        extract_key_sents(i)

    only_int_ranks = [rank for rank in ranks if isinstance(rank, int)]
    # print(predictions)
    print("====================================== Key sentences results =============================================")
    print("Key sentences ranks: ", ranks)
    print("Key sentecces unpredicted: ", unpredicted)
    print("Key sentences predicted: %d" % len(only_int_ranks))
    print("Number of documents whose patent terms extracted ok (answers within top 5 rank): %d" % (len([r for r in only_int_ranks if r <= 5])))
    print("Key sentences average rank: %f" % (sum(only_int_ranks) / len(only_int_ranks)))
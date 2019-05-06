import spacy
import os
from spacy import displacy
from collections import Counter
import en_core_web_sm

data_dir = "..\\ivan_data\\merged"
file_list = os.listdir(data_dir)


nlp = en_core_web_sm.load()
text = None

def read_data(data_dir, data_file_name):
    return open(os.path.join(data_dir, data_file_name), "r", encoding="cp950").read()


def pipeline(file_id):
    try:
        raw_text = read_data(data_dir, file_list[file_id])
    except:
        raise Exception("File not found!")
    fo = open("spacy_ner_out_{}".format(file_id), "w")
    text = nlp(raw_text)
    for token in text:
        print(token.text + "\t" + token.tag_)
    for ent in text.ents:
        fo.write(ent.text + "\t" + ent.label_ + "\n")
    fo.close()
    html_o = open("spacy_ner.html", "w")
    html_o.write(displacy.render(text, jupyter=False, style='ent'))
    html_o.close()

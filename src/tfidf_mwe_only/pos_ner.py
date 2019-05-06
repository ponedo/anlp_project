import spacy
import plac
import random
import pickle
from pathlib import Path
from spacy.util import minibatch, compounding
from spacy.tokens import Doc
from utils import *


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

# training data
TRAIN_DATA = [
    ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
    ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
]

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


def get_train_data():
    pass


def pipeline(file_id):
    raw_text = read_annotated_file(file_id)
    file_name = annotated_file_list[file_id]
    print(file_name)

    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    # if need to modify SpaCy, do it here
    
    sents = [sent.strip() for sent in raw_text.split("\n") if sent]

    f_data_label = open(os.path.join(pos_ner_tagged_data_dir, file_name) + ".pk", "wb")
    # f_text = open(os.path.join(pos_ner_tagged_data_dir, file_name), "w", encoding="cp950")
    
    sents_data_label = []

    for sent in sents:
        splited = sent.split("\t")
        sent = splited[0]

        if len(splited) > 1:
            labels = splited[1:]
        else:
            labels = []
        
        doc = nlp(sent)
        sent_data = [(t.text, t.tag_, t.ent_type_, t.ent_iob_) for t in doc]
        sent_label = []
        for i in range(len(doc)):
            if str(i) in labels:
                sent_label.append(1)
            else:
                sent_label.append(0)
        
        sents_data_label.append((sent_data, sent_label))

        # f_text.write(" ".join([str((t.text, t.tag_, t.ent_type_, t.ent_iob_)) for t in doc]))
        # f_text.write("\n")
        # f_text.write(" ".join([str(ent) for ent in doc.ents]))
        # f_text.write("\n__NEW_SENT__\n")
            
    pickle.dump(sents_data_label, f_data_label)

    f_data_label.close()
    # f_text.close()


if __name__ == "__main__":
    # pipeline(7)
    for i in range(len(preprocessed_file_list)):
        pipeline(i)

    # plac.call(main)

    # Expected output:
    # Entities [('Shaka Khan', 'PERSON')]
    # Tokens [('Who', '', 2), ('is', '', 2), ('Shaka', 'PERSON', 3),
    # ('Khan', 'PERSON', 1), ('?', '', 2)]
    # Entities [('London', 'LOC'), ('Berlin', 'LOC')]
    # Tokens [('I', '', 2), ('like', '', 2), ('London', 'LOC', 3),
    # ('and', '', 2), ('Berlin', 'LOC', 3), ('.', '', 2)]
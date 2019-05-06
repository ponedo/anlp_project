from gensim.models import Word2Vec
from utils import *


sg = 0
size = word_emb_dim
window = 5
min_count = 3
negative = 3
sample = 0.001
hs = 1


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
    
    def __iter__(self):
        for file_name in os.listdir(self.dirname):
            print(file_name)
            file_path = os.path.join(word2vec_data_dir, file_name)
            with open(file_path, "r", encoding="cp950") as f:
                for line in f:
                    line = line.strip().split()
                    if line:
                        yield line


if __name__ == '__main__':
    docs = MySentences(word2vec_data_dir)

    model = Word2Vec(
        docs,
        sg=sg,
        size=size,
        window=window,
        min_count=min_count,
        negative=negative,
        sample=sample,
        hs=hs)

    model_name = \
        'minc' + str(min_count) + \
        '_nega' + str(negative) + \
        '_sg' + str(sg) + \
        '_hs' + str(hs) + \
        '.txt'

    model.wv.save_word2vec_format(
        os.path.join(word2vec_dir, model_name),
        binary=False)

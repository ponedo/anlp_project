# Acknowledgement: anlp19/8.neural/CNN.ipynb


import keras
import os
import numpy as np
from sklearn import preprocessing
from keras.layers import Dense, Input, Embedding, GlobalMaxPooling1D, Conv1D, Concatenate, Dropout
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from utils import save_stuff
from step6_get_sent_dataset import load_dataset_for_CNN


train_ids = list(range(0, 100)) + list(range(140, 140))
valid_ids = list(range(100, 140))


def load_embeddings(filename, max_vocab_size):
    vocab={}
    embeddings=[]

    with open(filename, encoding="utf-8") as file:
        cols=file.readline().split(" ")
        num_words=int(cols[0])
        size=int(cols[1])
        embeddings.append(np.zeros(size))  # 0 = 0 padding if needed
        embeddings.append(np.zeros(size))  # 1 = UNK
        vocab["_0_"]=0
        vocab["_UNK_"]=1
        
        for idx, line in enumerate(file):
            if idx+2 >= max_vocab_size:
                break
            cols=line.rstrip().split(" ")
            val=np.array(cols[1:])
            word=cols[0]
            embeddings.append(val)
            vocab[word]=idx+2

    return np.array(embeddings), vocab, size


embeddings, vocab, embedding_size=load_embeddings("../ivan_data/glove.42B.300d.50K.w2v.txt", 100000)


def get_word_ids(docs, vocab, max_length=1000):
    doc_ids=[]
    for doc in docs:
        wids=[]
        for token in doc[:max_length]:
            val = vocab[token.lower()] if token.lower() in vocab else 1
            wids.append(val)

        # pad each document to constant width
        for i in range(len(wids),max_length):
            wids.append(0)

        doc_ids.append(wids)

    return np.array(doc_ids)


def cnn_sequential(embeddings, vocab_size, word_embedding_dim):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=word_embedding_dim, weights=[embeddings], trainable=False))
    model.add(Conv1D(filters=50, kernel_size=2, strides=1, padding="same", activation="tanh", name="CNN_bigram"))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model


def cnn(embeddings, vocab_size, word_embedding_dim):
    word_sequence_input = Input(shape=(None,), dtype='int32')
    word_embedding_layer = Embedding(vocab_size,
                                    word_embedding_dim,
                                    weights=[embeddings],
                                    trainable=False)
    
    embedded_sequences = word_embedding_layer(word_sequence_input)
    
    cnn2=Conv1D(filters=10, kernel_size=2, strides=1, padding="same", activation="tanh", name="CNN_bigram")(embedded_sequences)
    cnn3=Conv1D(filters=5, kernel_size=3, strides=1, padding="same", activation="tanh", name="CNN_trigram")(embedded_sequences)
    cnn4=Conv1D(filters=3, kernel_size=4, strides=1, padding="same", activation="tanh", name="CNN_4gram")(embedded_sequences)

    # max pooling over all words in the document
    maxpool2=GlobalMaxPooling1D()(cnn2)
    maxpool3=GlobalMaxPooling1D()(cnn3)
    maxpool4=GlobalMaxPooling1D()(cnn4)

    x=Concatenate()([maxpool2, maxpool3, maxpool4])

    x=Dropout(0.2)(x)
    x=Dense(10)(x)
    x=Dense(1, activation="sigmoid")(x)

    model = Model(inputs=word_sequence_input, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    
    return model


if __name__ == "__main__":

    _, train_data, train_labels = zip(*load_dataset_for_CNN(train_ids))
    _, valid_data, valid_labels = zip(*load_dataset_for_CNN(valid_ids))
    # _, test_data, test_labels = zip(*load_dataset_for_CNN(valid_ids))

    trainX, trainY = get_word_ids(train_data, vocab, max_length=75), train_labels
    devX, devY = get_word_ids(valid_data, vocab, max_length=75), valid_labels

    # le = preprocessing.LabelEncoder()
    # le.fit(trainY)
    # Y_train=np.array(le.transform(trainY))
    # Y_dev=np.array(le.transform(devY))
    Y_train = np.array(trainY)
    Y_dev = np.array(devY)

    # cnn_sequential_model = cnn_sequential(embeddings, len(vocab), embedding_size)
    # print(cnn_sequential_model.summary())

    cnn_functional_model = cnn(embeddings, len(vocab), embedding_size)
    print (cnn_functional_model.summary())

    early_stopping = EarlyStopping(monitor='val_loss',
        min_delta=0, patience=5, verbose=0, mode='auto')

    checkpoint = ModelCheckpoint("CNN.hdf5", monitor='val_loss', verbose=0, save_best_only=True, mode='min')

    model = cnn_functional_model
    model.fit(trainX, Y_train, 
                validation_data=(devX, Y_dev),
                epochs=1, batch_size=128, 
                callbacks=[checkpoint, early_stopping])
    
    save_stuff(model, "CNN")
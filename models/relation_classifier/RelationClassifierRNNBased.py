import utils.data as datautils


import re
from unidecode import unidecode
from keras import Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LambdaCallback
from keras.engine import Model
from keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense, Dropout, Activation
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import constants as c
import numpy as np

from models.relation_classifier import LSTM_SIZE, EMBEDDING_SIZE, DROPOUT, BATCH_SIZE, MAX_EPOCHS
from utils.dependency_parser import get_spacy_parser
import pickle


class RelationClassifierRNNBased():
    def __init__(self,
                 lstm_size=LSTM_SIZE,
                 embedding_size=EMBEDDING_SIZE,
                 dropout=DROPOUT,
                 batch_size=BATCH_SIZE,
                 max_epochs=MAX_EPOCHS
                 ):
        self.w2idx = {c.PAD_TAG: 0, c.UNK_TAG: 1}
        self.idx2w = {0: c.PAD_TAG, 1: c.UNK_TAG}

        self.rel2idx = {c.NIL_TAG:0}
        self.idx2rel = {0:c.NIL_TAG}
        self.MAX_LENGTH = 35

        self.lstm_size = lstm_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.max_epochs = max_epochs



    def make_vocab(self, sents):
        parser = get_spacy_parser()
        for s in sents:
            cleaned_s = datautils.clean_question(s)
            doc = parser.tokenizer(cleaned_s)
            for t in doc:
                if not t.text in self.w2idx:
                    new_idx = len(self.w2idx)
                    self.w2idx[t.text] = new_idx
                    self.idx2w[new_idx] = t.text
                if not t.text.lower() in self.w2idx:
                    new_idx = len(self.w2idx)
                    self.w2idx[t.text.lower()] = new_idx
                    self.idx2w[new_idx] = t.text.lower()

    def to_word_list(self, vect):
        words = list(map(lambda i: self.idx2w[i], vect))
        return words

    def to_tags(self, vect):
        tags = c.entity_tags[vect.argmax()]
        return tags


    def fit(self, X,Y, dev=None):
        # early stopping callback for stop the training when needed
        earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

        # reduce learning rate callback for optimize on plateau
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=0, verbose=1, mode='auto',
                                      epsilon=0.0001, cooldown=0, min_lr=0)

        def printer(x):
            print(x)

        print_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: printer(logs),
            on_train_end=lambda logs: printer(logs)
        )

        self.model.fit([X], Y, epochs=self.max_epochs, batch_size=self.batch_size,
                       shuffle=True,
                       validation_data=dev,
                       verbose=1,
                       callbacks=[print_callback, reduce_lr, earlyStopping],
                       )

    def train(self, sents, concepts, dev=None):
        X, Y = self.make_XY(sents, concepts)
        if dev:
            X_dev, Y_dev = self.make_XY(dev[0], dev[1])
        else:
            (X_dev, Y_dev) = (None, None)
        self.model = self.get_model()
        self.fit(X,Y,dev=(X_dev, Y_dev))


    def test(self, sents, relations):
        X,Y = self.make_XY(sents, relations)
        Y_pred = self.model.predict(X)

        total = len(X)
        correct = 0
        prediction_per_relation = {k:[0,0] for k in c.RELATIONS}


        for x,yp,yt, sent, r in zip(X, Y_pred, Y, sents, relations):
            no_pads = [t for t in x if t!=0]
            true_len = len(no_pads)
            pred_tags = self.idx2rel[yp.argmax()]
            true_tags = self.idx2rel[yt.argmax()]

            prediction_per_relation[true_tags][1] += 1
            if true_tags==pred_tags:
                correct+=1
                prediction_per_relation[true_tags][0] += 1
            else:
                print("="*50)
                print(sent)
                print(r)
                print(self.to_word_list(x)[:true_len])
                print(pred_tags)
                print(true_tags)
                print("=" * 50)

        print(total, correct)
        print(correct/total)
        print("accuracy per relation: ")


        print("Relation\tSupport\tCorrect\tAcc.")
        for k, (cor, tot) in prediction_per_relation.items():
            print("\t".join([k, str(tot), str(cor), str(cor/tot)]))

        print("\t".join(["Total", str(total), str(correct), str(correct/total)]))

    def get_model(self):
        main_input = Input(shape=(self.MAX_LENGTH,), name="main_input")

        embedded = Embedding(len(self.w2idx),self.embedding_size,mask_zero=True)(main_input)

        blstm_l1 = Bidirectional(LSTM(
            self.lstm_size,
            return_sequences=True)
        )(embedded)

        blstm_l2 = Bidirectional(LSTM(
            self.lstm_size,
            return_sequences=False)
        )(blstm_l1)


        output_l = Dense(len(self.rel2idx), use_bias=True)(blstm_l2)
        dropout_out_l = Dropout(0.1)(output_l)
        activation = Activation(activation="softmax")(dropout_out_l)

        model = Model(inputs=[main_input], outputs=[activation])

        # learning rate = 0.001
        opt = RMSprop()
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        return model

    def make_x(self, sent):
        parser = get_spacy_parser()
        question = datautils.clean_question(sent)

        # tokenize the question string
        q_parsed = parser(question)
        q_vect = list(map(lambda t: self.w2idx[t.text] if t.text in self.w2idx else self.w2idx[c.UNK_TAG], q_parsed))
        pad_length = self.MAX_LENGTH - len(q_vect)
        nil_X = 0
        x = (q_vect + (pad_length) * [nil_X])
        return np.array(x)


    def make_XY(self, sents, relations):
        """

        :param sents: a list of strings
        :param relations: a list of relation strings
        :return: X, Y, where:
            - X is a list of one-hot encoded sentences (eventually padded), but not vectorized
                i.e. a np.Array of shape (len(sents), MAX_LENGTH)
            - Y is the list of one-hot encoded relations.
        """
        X, Y = [], []
        for sent, rel in zip(sents, relations):

            x = self.make_x(sent)

            if not rel in self.rel2idx:
                new_idx = len(self.rel2idx)
                self.rel2idx[rel]=new_idx
                self.idx2rel[new_idx]=rel

            y = datautils.to_ohenc(self.rel2idx[rel], 17)

            X.append(np.array(x))
            Y.append(np.array(y))
        return np.array(X),np.array(Y)

    def predict(self, query, candidate_relations=[]):
        x = self.make_x(query)

        Y_pred = self.model.predict(np.array([x]))

        Y_pred = enumerate(Y_pred[0])
        Y_pred = sorted(Y_pred, key=lambda x: -x[1])


        relation = self.idx2rel[Y_pred[0][0]]

        # Actually, the mapping relation 2 domain are not so good.
        # For example, for "Geography and places" there missing the "PLACE" relation, which I think it is necessary.
        # However, the implementation is below.

        # if candidate_relations and relation not in candidate_relations:
        #     Y_pred = list(filter(lambda x: x[0]!=self.rel2idx[c.NIL_TAG]  and self.idx2rel[x[0]] in candidate_relations, Y_pred))
        #     Y_pred = sorted(Y_pred, key=lambda x: -x[1])
        #     if len(Y_pred)==0:
        #         relation =""
        #     else:
        #         relation = self.idx2rel[Y_pred[0][0]]

        return relation

    def save(self, dirpath):
        import os
        os.system("rm -rf " + dirpath)
        os.mkdir(dirpath)
        temp = self.get_model()
        self.model.save(dirpath+"/model")
        self.model = None
        import pickle
        pickle.dump(self, open(dirpath+"/obj", "wb"))
        self.model = temp

    @staticmethod
    def load(dirpath):
        obj = pickle.load(open(dirpath+"/obj", "rb"))
        model = load_model(dirpath+"/model")
        obj.model = model
        return obj


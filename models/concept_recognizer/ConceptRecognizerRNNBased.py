from keras import Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LambdaCallback
from keras.engine import Model
from keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense, Dropout, Activation, add
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import constants as c
import utils.data as datautils
import numpy as np
from keras.models import load_model
import utils.misc as misc

from models.concept_recognizer import LSTM_SIZE, EMBEDDING_SIZE, DROPOUT, MAX_EPOCHS, BATCH_SIZE, MAX_LENGTH
from utils.dependency_parser import get_spacy_parser
class ConceptRecognizerRNNBased():

    def __init__(self,
                 lstm_size=LSTM_SIZE,
                 embedding_size=EMBEDDING_SIZE,
                 dropout=DROPOUT,
                 batch_size=BATCH_SIZE,
                 max_epochs=MAX_EPOCHS
                 ):

        self.w2idx = {c.PAD_TAG:0,c.UNK_TAG:1}
        self.idx2w = {0:c.PAD_TAG, 1: c.UNK_TAG}

        self.MAX_LENGTH = MAX_LENGTH

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

    def to_tag_sequence(self, seq):
        return list(map(lambda x: self.to_tags(x), seq))


    def fit(self, X,Y, dev=None):
        # early stopping callback for stop the training when needed
        earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

        # reduce learning rate callback for optimize on plateau
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=0, verbose=1, mode='auto',
                                      epsilon=0.0001, cooldown=0, min_lr=0)


        def printer(x):
            print(x)

        print_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: printer(logs),
            on_train_end=lambda logs: printer(logs)
        )

        self.model.fit(X, Y, epochs=self.max_epochs, batch_size=self.batch_size,
                       shuffle=True,
                       validation_data=dev,
                       verbose=1,
                       callbacks=[print_callback, reduce_lr, earlyStopping],
                       )

    def train(self, sents_relations, concepts, dev=None):
        """
        X_train:
            list of (question, relation)
        Y_train:
            list of (c1::babelnet_id, c2::babelnet_id)

        """

        X, Y = self.make_XY(sents_relations, concepts)
        if dev:
            X_dev, Y_dev = self.make_XY(dev[0], dev[1])
        else:
            (X_dev, Y_dev) = (None, None)
        self.model = self.get_model()
        self.fit(X,Y,dev=(X_dev, Y_dev))



    def test(self, sents_relations, concepts):
        total =  len(sents_relations)
        correct = 0
        tags_tot = 0
        tags_correct = 0
        prediction_per_relation = {k:[0,0] for k in c.RELATIONS}

        X,Y = self.make_XY(sents_relations, concepts)
        Y_pred = self.model.predict(X)


        new_sents_relations = zip(*X)
        for (x, rel), yp,yt, sent, cs in zip(new_sents_relations, Y_pred, Y, sents_relations, concepts):
            no_pads = [t for t in x if t!=0]
            true_len = len(no_pads)
            pred_tags = list(map(self.to_tags, yp))[:true_len]
            true_tags = list(map(self.to_tags, yt))[:true_len]
            tags_tot+=true_len
            tags_correct+=sum([p==t for p,t in zip(pred_tags,true_tags)])

            rel = c.RELATIONS[rel.argmax()]
            prediction_per_relation[rel][1]+=1

            if true_tags==pred_tags:
                correct+=1
                prediction_per_relation[rel][0] += 1

            else:
                print("="*50)
                print(rel)
                print(sent)
                print(cs)
                print(self.to_word_list(x)[:true_len])
                print(pred_tags)
                print(true_tags)
                print("=" * 50)

        print("#sentences: ", total, " #correct", correct, " acc: ", correct/total)

        print("Relation\tSupport\tCorrect\tAcc.")
        for k, (cor, tot) in prediction_per_relation.items():
            print("\t".join([k, str(tot), str(cor), str(cor / tot)]))

        print("\t".join(["Total", str(total), str(correct), str(correct / total)]))
        print("#tags: ", tags_tot, " #correct", tags_correct, " acc: ", tags_correct/tags_tot)




    def get_model(self):
        main_input = Input(shape=(self.MAX_LENGTH,), name="main_input")

        embedded = Embedding(len(self.w2idx),self.embedding_size,mask_zero=True)(main_input)

        relation_input = Input(shape=(len(c.RELATIONS),), name="relation_input")
        relation_matrix = Dense(self.embedding_size, input_shape=(None, len(c.RELATIONS)))(relation_input)

        merge_l = add([embedded, relation_matrix])


        blstm_l1 = Bidirectional(LSTM(
            self.lstm_size,
            return_sequences=True)
        )(merge_l)

        blstm_l2 = Bidirectional(LSTM(
            self.lstm_size,
            return_sequences=True)
        )(blstm_l1)


        output_l = TimeDistributed(Dense(len(c.entity_tags), use_bias=True))(blstm_l2)
        dropout_out_l = Dropout(self.dropout)(output_l)
        activation = Activation(activation="softmax")(dropout_out_l)

        model = Model(inputs=[main_input, relation_input], outputs=[activation])

        # learning rate = 0.001
        opt = RMSprop()
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        return model


    def make_x(self, sent, relation):
        parser = get_spacy_parser()
        question = datautils.clean_question(sent)
        q_parsed = parser(question)
        q_vect = list(
            map(lambda t: self.w2idx[t.text] if t.text in self.w2idx else self.w2idx[c.UNK_TAG], q_parsed))
        pad_length = self.MAX_LENGTH - len(q_vect)
        nil_X = 0
        x = (q_vect + (pad_length) * [nil_X])
        x= np.array(x)


        x_rel = datautils.to_ohenc(c.RELATIONS.index(relation), len(c.RELATIONS))
        return np.array([x, x_rel])

    def make_XY(self, sents_relations, concepts):
        """

        :param sents_relations: list of pairs (question, relations)
        :param concepts: list of pairs (c1, c2)
        :return:
        """
        parser = get_spacy_parser()
        X, Y = [], []
        for (sent, relation), c_list in zip(sents_relations, concepts):
            x = self.make_x(sent, relation)

            question = datautils.clean_question(sent)
            q_parsed = parser(question)
            c1, c2 = datautils.clean_concept(c_list[0]), datautils.clean_concept(c_list[1])

            # find the indexes of the concept mentions
            c1_idx = question.find(c1)
            assert c1_idx != -1
            assert question[c1_idx:c1_idx + len(c1)] == c1

            # print((question[:c1_idx] + "#"*len(c1) + question[c1_idx+len(c1):]))
            c2_idx = (question[:c1_idx] + "#"*len(c1) + question[c1_idx+len(c1):]).find(c2)
            if c2_idx != -1:
                assert question[c2_idx:c2_idx + len(c2)] == c2

            # iterate over tokens of the question
            # if the index falls into concept mentions indexes, then it is a concept right or left)
            tags = list(map(lambda t:
                            datautils.to_ohenc(c.entity_tags.index(c.LEFT_ENT_TAG), len(c.entity_tags))
                                if (t.idx >= c1_idx and t.idx + len(t) <= c1_idx + len(c1))
                                else
                                    datautils.to_ohenc(c.entity_tags.index(c.RIGHT_ENT_TAG), len(c.entity_tags))
                                    if  (c2_idx!=-1 and t.idx >= c2_idx and t.idx + len(t) <= c2_idx + len(c2))
                                else datautils.to_ohenc(c.entity_tags.index(c.N_ENT_TAG), len(c.entity_tags)), q_parsed)
                        )



            nil_Y = datautils.to_ohenc(c.entity_tags.index(c.NIL_TAG), len(c.entity_tags))

            pad_length = self.MAX_LENGTH - len(tags)
            y = (tags + ((pad_length) * [nil_Y]))

            X.append(np.array(x))
            Y.append(np.array(y))

        X = [np.array(t) for t in zip(*X)]
        # at the end, X is a list of two arrays:
        #     the 1st is a list of sentences (in indexed forms)
        #     the 2nd is a list of relation representation
        # Y is a list of samples, each of them a list of tags
        return X,np.array(Y)


    def predict(self, sentence, relation):
        """
        Predict the tags for each token in sentence.
        :param sentence: the sentence to tag with conccept tags (see constants.(LEFT|RIGHT)_CONCEPT_TAG
        :param relation: the current relation
        :return: a dictionary:
            {concept_tag : list of tokenized concept mentions}

            e.g.
            Is the University of Rome in Rome?
            {
                "l": [ ["University", "of", "Rome"] ],
                "r": [ ["Rome"] ]
             }

        """
        x = self.make_x(sentence, relation)
        X = [np.array(t) for t in zip(*[x])]
        Y_pred = self.model.predict(X)
        no_pads = [t for t in x[0] if t != self.w2idx[c.PAD_TAG]]
        true_len = len(no_pads)

        word_list = [t.text for t in get_spacy_parser().tokenizer(sentence)]
        tag_sequence = self.to_tag_sequence(Y_pred[0])[:true_len]


        type2concepts = misc.merge_concept_tags(word_list, tag_sequence)

        return type2concepts



    def save(self, dirpath):
        import os
        os.system("rm -rf " + dirpath)
        os.mkdir(dirpath)
        self.model.save(dirpath+"/model")
        self.model = None
        import pickle
        pickle.dump(self, open(dirpath+"/obj", "wb"))

    @staticmethod
    def load(dirpath):
        import pickle
        obj = pickle.load(open(dirpath+"/obj", "rb"))
        model = load_model(dirpath+"/model")
        obj.model = model
        return obj


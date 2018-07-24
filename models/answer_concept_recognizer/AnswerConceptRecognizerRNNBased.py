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
class AnswerConceptRecognizerRNNBased():
    """ It is like the ConceptRecognizer, but instead of questions, we use answers"""
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

        # self.lstm_size = 32
        # self.embedding_size = 64
        # self.dropout = 0.0
        # self.batch_size = 32
        # self.max_epochs = 3


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

    def train(self, questions_answers_relations, concepts, dev=None):
        X, Y = self.make_XY(questions_answers_relations, concepts)
        if dev:
            X_dev, Y_dev = self.make_XY(dev[0], dev[1])
        else:
            (X_dev, Y_dev) = (None, None)
        self.model = self.get_model()
        self.fit(X,Y,dev=(X_dev, Y_dev))



    def test(self, questions_answers_relations, concepts):
        total = len(questions_answers_relations)
        correct = 0
        tags_tot = 0
        tags_correct = 0
        prediction_per_relation = {k:[0,0] for k in c.RELATIONS}

        X,Y = self.make_XY(questions_answers_relations, concepts)
        Y_pred = self.model.predict(X)

        # the question input layer is an older version: you can ignore the question parts
        new_questions_answers_relations = zip(*X)
        for (answer, rel), yp,yt, sent, cs in zip(new_questions_answers_relations, Y_pred, Y, questions_answers_relations, concepts):
            no_pads = [t for t in answer if t!=self.w2idx[c.PAD_TAG]]
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
                aaa = self.to_word_list(answer)
                print(aaa[:true_len])
                print(pred_tags)
                print(true_tags)
                print("=" * 50)

        print("Relation\tSupport\tCorrect\tAcc.")
        for k, (cor, tot) in prediction_per_relation.items():
            print("\t".join([k, str(tot), str(cor), str(cor / tot)]))

        print("\t".join(["Total", str(total), str(correct), str(correct / total)]))
        print("#tags: ", tags_tot, " #correct", tags_correct, " acc: ", tags_correct/tags_tot)


    def get_model(self):

        # answer input
        answer_input = Input(shape=(self.MAX_LENGTH,), name="main_input")
        embedded = Embedding(len(self.w2idx),self.embedding_size,mask_zero=True)(answer_input)

        relation_input = Input(shape=(len(c.RELATIONS),), name="relation_input")
        relation_matrix = Dense(self.embedding_size, input_shape=(None, len(c.RELATIONS)))(relation_input)

        # the question input layer is an older version: you can ignore it

        # question_input = Input(shape=(self.MAX_LENGTH,), name="question_input")
        # question_embedded = Embedding(len(self.w2idx),self.embedding_size,mask_zero=True)(question_input)
        # question_lstm_input = LSTM(self.lstm_size, return_sequences=False, name="question_lstm")(question_embedded)
        # question_matrix = Dense(self.embedding_size)(question_lstm_input)


        # merge_l = add([embedded, relation_matrix, question_input])
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

        model = Model(inputs=[answer_input, relation_input], outputs=[activation])

        # learning rate = 0.001
        opt = RMSprop()
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        return model

    # the question input layer is an older version: you can ignore the question parts
    # def make_x(self, question, answer, relation):
    def make_x(self, answer, relation):
        parser = get_spacy_parser()

        # question = datautils.clean_question(question)
        # q_parsed = parser(question)
        # q_vect = list(
        #     map(lambda t: self.w2idx[t.text] if t.text in self.w2idx else self.w2idx[c.UNK_TAG], q_parsed))
        # pad_length = self.MAX_LENGTH - len(q_vect)
        # nil_X = 0
        # x_question = (q_vect + (pad_length) * [nil_X])
        # x_question= np.array(x_question)

        answer = datautils.clean_question(answer)
        a_parsed = parser(answer)
        a_vect = list(
            map(lambda t: self.w2idx[t.text] if t.text in self.w2idx else self.w2idx[c.UNK_TAG], a_parsed))
        pad_length = self.MAX_LENGTH - len(a_vect)
        nil_X = 0
        x_answer = (a_vect + (pad_length) * [nil_X])
        x_answer = np.array(x_answer)

        x_rel = datautils.to_ohenc(c.RELATIONS.index(relation), len(c.RELATIONS))
        return np.array([x_answer, x_rel])

    def make_XY(self, questions_answers_relations, concepts):
        parser = get_spacy_parser()
        X, Y = [], []
        for (question, answer, relation), c_list in zip(questions_answers_relations, concepts):
            # x = self.make_x(question, answer, relation)
            x = self.make_x(answer, relation)
            answer = datautils.clean_question(answer)
            a_parsed = parser(answer)
            c1, c2 = datautils.clean_concept(c_list[0]), datautils.clean_concept(c_list[1])

            # the question input layer is an older version: you can ignore the question parts
            # assert c1_idx != -1
            # assert question[c1_idx:c1_idx + len(c1)] == c1

            # print((question[:c1_idx] + "#"*len(c1) + question[c1_idx+len(c1):]))
            # c2_idx = (question[:c1_idx] + "#"*len(c1) + question[c1_idx+len(c1):]).find(c2)

            c1_idx = answer.find(c1)
            c2_idx = answer.find(c2)

            tags = list(map(lambda t:
                            datautils.to_ohenc(c.entity_tags.index(c.RIGHT_ENT_TAG), len(c.entity_tags))
                            if (c2_idx != -1 and t.idx >= c2_idx and t.idx + len(t) <= c2_idx + len(c2))
                                else
                                    datautils.to_ohenc(c.entity_tags.index(c.LEFT_ENT_TAG), len(c.entity_tags))
                                    if (c1_idx != -1 and t.idx >= c1_idx and t.idx + len(t) <= c1_idx + len(c1))
                                else datautils.to_ohenc(c.entity_tags.index(c.N_ENT_TAG), len(c.entity_tags)),

                            a_parsed)
                        )
            print(c1,";", c2)
            print(a_parsed)
            print(self.to_tag_sequence(tags))

            nil_Y = datautils.to_ohenc(c.entity_tags.index(c.NIL_TAG), len(c.entity_tags))

            pad_length = self.MAX_LENGTH - len(tags)
            y = (tags + ((pad_length) * [nil_Y]))

            X.append(np.array(x))
            Y.append(np.array(y))

        X = [np.array(t) for t in zip(*X)]
        return X,np.array(Y)


    # def predict(self, sentence, answer, relation):
    #     x = self.make_x(sentence, answer, relation)
    def predict(self,answer, relation):
        """See the ConceptRecognizer classifier, they are similar"""
        x = self.make_x(answer, relation)
        X = [np.array(t) for t in zip(*[x])]
        Y_pred = self.model.predict(X)
        no_pads = [t for t in x[0] if t != self.w2idx[c.PAD_TAG]]
        true_len = len(no_pads)

        word_list = [t.text for t in get_spacy_parser().tokenizer(answer)]
        tag_sequence = self.to_tag_sequence(Y_pred[0])[:true_len]

        type2concepts = misc.merge_concept_tags(word_list, tag_sequence)
        print(type2concepts)

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

